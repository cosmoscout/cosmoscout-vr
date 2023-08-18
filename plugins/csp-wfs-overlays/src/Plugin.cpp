////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "FeatureRenderer.hpp"
#include "PointRenderer.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/ThreadPool.hpp"
#include "logger.hpp"

#include <iostream>

#include <VistaTools/tinyXML/tinyxml.h>

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

#include <thread>
#include <chrono>

#include <regex>

#include <boost/filesystem.hpp>

#include "../../../src/cs-scene/CelestialSurface.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::wfsoverlays::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::wfsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

// bind the components of the Settings struct (from the Plugin class)
// defined at the header with the elements at the simple_desktop.json
//----------------------------------------------------------------------
    
void from_json(nlohmann::json const& j, Plugin::Settings& o) {       // here, it is (m)Enabled, (m)Wfs because the
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);         // simple_desktop.json file has no constructors
  cs::core::Settings::deserialize(j, "wfs", o.mWfs); 
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "wfs", o.mWfs);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// from_json for all the different structs defined at the WFSTypes.hpp
//--------------------------------------------------------------------

void from_json(const nlohmann::json& j, std::shared_ptr<GeometryBase>& o) {
  
  // for datasets containing "null" geometry (e.g. DWD -> dwd:Autowarn_Vorhersage)
  if (j.is_null()) {
    o = nullptr;
    return;
  }

  std::string type; 
  cs::core::Settings::deserialize(j, "type", type); 

  if (type=="Point") {
    PointCoordinates coordinates; 
    cs::core::Settings::deserialize(j, "coordinates", coordinates); 
    o = std::make_shared<Point> (coordinates);
  } 

  if (type=="MultiPoint") {
    LineStringCoordinates coordinates; 
    cs::core::Settings::deserialize(j, "coordinates", coordinates); 
    o = std::make_shared<MultiPoint> (coordinates);
  } 

  if (type=="LineString") {
    LineStringCoordinates coordinates; 
    cs::core::Settings::deserialize(j, "coordinates", coordinates); 
    o = std::make_shared<LineString> (coordinates);
  } 

  if (type=="MultiLineString") {
    PolygonCoordinates coordinates; 
    cs::core::Settings::deserialize(j, "coordinates", coordinates); 
    o = std::make_shared<MultiLineString> (coordinates);
  } 

  if (type=="Polygon") {
    PolygonCoordinates coordinates; 
    cs::core::Settings::deserialize(j, "coordinates", coordinates); 
    o = std::make_shared<Polygon> (coordinates);
  } 

  if (type=="MultiPolygon") {
    MultiPolygonCoordinates coordinates; 
    cs::core::Settings::deserialize(j, "coordinates", coordinates); 
    o = std::make_shared<MultiPolygon> (coordinates);
  } 
} 

// here it would be the Properties1

void from_json(const nlohmann::json& j, Feature& o) { 
  cs::core::Settings::deserialize(j, "type", o.type); 
  cs::core::Settings::deserialize(j, "id", o.id);
  cs::core::Settings::deserialize(j, "geometry", o.geometry); // Struct
  cs::core::Settings::deserialize(j, "geometry_name", o.geometry_name);
  cs::core::Settings::deserialize(j, "properties", o.properties); // Prop
  //cs::core::Settings::deserialize(j, "bbox", o.bbox); 
} 

// here it would be the Properties2

void from_json(const nlohmann::json& j, CRS& o) { 
  cs::core::Settings::deserialize(j, "type", o.type); 
  //cs::core::Settings::deserialize(j, "properties", o.properties);  // Prop
} 

void from_json(const nlohmann::json& j, WFSFeatureCollection& o) { 
  cs::core::Settings::deserialize(j, "type", o.type); 
  cs::core::Settings::deserialize(j, "features", o.features); // Vector
  cs::core::Settings::deserialize(j, "totalFeatures", o.totalFeatures);
  cs::core::Settings::deserialize(j, "numberMatched", o.numberMatched);
  cs::core::Settings::deserialize(j, "numberReturned", o.numberReturned); 
  cs::core::Settings::deserialize(j, "timeStamp", o.timeStamp);
  cs::core::Settings::deserialize(j, "crs", o.crs);
  //cs::core::Settings::deserialize(j, "bbox", o.bbox); // Array 
} 

////////////////////////////////////////////////////////////////////////////////////////////////////

// now the from_json for the Describe Feature Type
//------------------------------------------------

void from_json(const nlohmann::json& j, Property& o) {
  cs::core::Settings::deserialize(j, "name", o.name);
  cs::core::Settings::deserialize(j, "maxOccurs", o.maxOccurs);
  cs::core::Settings::deserialize(j, "minOccurs", o.minOccurs);
  cs::core::Settings::deserialize(j, "nillable", o.nillable);
  cs::core::Settings::deserialize(j, "type", o.type);
  cs::core::Settings::deserialize(j, "localType", o.type);
}

void from_json(const nlohmann::json& j, FeatureType& o) {
  cs::core::Settings::deserialize(j, "typeName", o.typeName);
  cs::core::Settings::deserialize(j, "properties", o.properties); // Vector
}

void from_json(const nlohmann::json& j, DescribeFeatureType& o) {
  cs::core::Settings::deserialize(j, "elementFormDefault", o.elementFormDefault);
  cs::core::Settings::deserialize(j, "targetNamespace", o.targetNamespace); 
  cs::core::Settings::deserialize(j, "targetPrefix", o.targetPrefix);
  cs::core::Settings::deserialize(j, "featureTypes", o.featureTypes); // Array
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// the function generates midPoints between the Cartesian components of a given
//  vector of InfoStructs (structIn) and returns a vector of dvec3 (renderingVector).  
//------------------------------------------------------------------------- 

std::vector<glm::dvec3> Plugin::generateMidPoint (std::vector <InfoStruct> const& structsIn, float threshold, 
                                                    glm::vec3 earthRadius, std::shared_ptr<const cs::scene::CelestialObject> earth, glm::vec3 featureColor) { 
  std::vector <InfoStruct> totalStruct;
  std::vector<glm::dvec3> renderingVector;    // the returned one

  // TODO: structsIn is already a STRUCT {v0, v1, v1, ... , vn-2, vn-2, vn-1}

  for (int i=0; i < structsIn.size()-1; i++) {  

    // the following definitions will just make the code syntax simpler 
    glm::dvec2 p1LongLatRadians = structsIn[i].longLatRadians;
    glm::dvec2 p2LongLatRadians = structsIn[i+1].longLatRadians;
    glm::dvec2 p1LongLatDegrees = structsIn[i].longLatDegrees;
    glm::dvec2 p2LongLatDegrees = structsIn[i+1].longLatDegrees;
    glm::dvec3 p1Cartesian      = structsIn[i].Cartesian;
    glm::dvec3 p2Cartesian      = structsIn[i+1].Cartesian;

    double distance = calculateDistance(structsIn[i], structsIn[i+1], earthRadius);

    if (distance > threshold) { 

      totalStruct.push_back(structsIn[i]);
      int numMidPoints = static_cast<int> (distance/threshold);
      double segmentLength = distance/threshold;

      for (int j=1; j <= numMidPoints; j++) {
        InfoStruct temporaryStruct;
        temporaryStruct.Cartesian = glm::mix(p1Cartesian,p2Cartesian,(segmentLength*j)/distance);   // without correct height
        temporaryStruct.longLatRadians = cs::utils::convert::cartesianToLngLat(temporaryStruct.Cartesian, earthRadius);
        
        correctHeight (structsIn[i], structsIn[i+1], temporaryStruct, earth);
        
        // correct the cartesian based on height
        temporaryStruct.Cartesian = cs::utils::convert::toCartesian(temporaryStruct.longLatRadians, earthRadius, temporaryStruct.overSurfaceHeight);  // with correct height
        totalStruct.push_back(temporaryStruct);
        totalStruct.push_back(temporaryStruct);   // TODO: as input already doubled, it would be: option1 = STRUCT con {v0, mids x2, v1, v1, mids x2, v2, v2, ... , vn-2, vn-2, mids x2}
      }
    }
    // in case the distance is so small that midPoints are not needed
    else {

      totalStruct.push_back(structsIn[i]);         // TODO: as input already doubled, it would be: option2 = STRUCT con {v0, v1, v1, v2, v2, ... , vn-2, vn-2}
    } 
  }
  
  totalStruct.push_back(structsIn[structsIn.size() - 1]);   // TODO: it would be STRUCT con: {option1 ,vn-1}   or   {option2 ,vn-1}
  
  // TODO: IF WE WANT TO USE THE INTERPOLATE.............. std::vector <InfoStruct> interpolatedStruct = Interpolation (totalStruct, 200.0, earthRadius, earth);

  for (int i=0; i<totalStruct.size(); i++) {    // TODO: IF WE WANT TO USE THE INTERPOLATE.............. interpolatedStruct    INSTEAD OF totalStruct
    renderingVector.push_back(totalStruct[i].Cartesian);    // TODO: IF WE WANT TO USE THE INTERPOLATE.............. interpolatedStruct    INSTEAD OF totalStruct
    renderingVector.push_back(featureColor);
  }

return renderingVector; // TODO: it would be {v0, rgb0, x2 (mid,rgb), v1, rgb1, v1, rgb1, x2 (mid,rgb), ... , vn-2, rgbn-2, vn-2, rgbn-2}
  }
  

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::correctHeight (InfoStruct const& struct1, InfoStruct const& struct2, InfoStruct& temporaryStruct,  
                                                    std::shared_ptr<const cs::scene::CelestialObject> earth) {
  // for the sake of simplicity
  bool p1Bool = struct1.heightComesFromJson;
  bool p2Bool = struct2.heightComesFromJson;
  double p1Height = struct1.overSurfaceHeight;
  double p2Height = struct2.overSurfaceHeight;

  if (p1Bool || p2Bool) {
    temporaryStruct.overSurfaceHeight = ((p1Height + p2Height)/2);
    temporaryStruct.heightComesFromJson = true;
  } 
  else {
    temporaryStruct.overSurfaceHeight = earth->getSurface()->getHeight(temporaryStruct.longLatRadians);
    temporaryStruct.heightComesFromJson = false;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// the function takes as an input the earthRadius and a couple of 
// points (in Cartesian), whose Great Circle distance will be returned.
//--------------------------------------------------------------------

double Plugin::calculateDistance(InfoStruct const& p1, InfoStruct const& p2, glm::vec3 earthRadius) { 
  
  // great circle distance using Haversine formula     
  double rootArgument = cos(p1.longLatRadians[1]) * cos(p2.longLatRadians[1]) * cos(p1.longLatRadians[0] - p2.longLatRadians[0]) + sin(p1.longLatRadians[1]) * sin(p2.longLatRadians[1]);
  double averageEarthRadius = (earthRadius[0] + earthRadius[1] + earthRadius[2])/3;
  double distanceHaversine = (averageEarthRadius + (p1.overSurfaceHeight+p2.overSurfaceHeight)/2)  * acos(rootArgument);

  // staight-line distance
  double distanceStraight = glm::distance(p1.Cartesian,p2.Cartesian);
  
  // comparison
  if (distanceStraight>distanceHaversine) {
    return distanceStraight;
  }
  else {
    return distanceHaversine;    
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////


double Plugin::calculateAngle (InfoStruct const& previousPoint, InfoStruct const& middlePoint, InfoStruct const& nextPoint) {

  glm::dvec3 vec1 = middlePoint.Cartesian - previousPoint.Cartesian;
  glm::dvec3 vec2 = nextPoint.Cartesian - middlePoint.Cartesian;

  double dotProduct = glm::dot(glm::normalize(vec1),glm::normalize(vec2));
  double angleRad =  glm::acos(dotProduct); // in radians
  double PI = glm::pi<double>();
  double angleDeg = angleRad * (180.0 / PI);
  return angleDeg;
}


////////////////////////////////////////////////////////////////////////////////////////////////////


std::vector<InfoStruct> Plugin::Interpolation (std::vector<InfoStruct> const& structsIn, double thresholdAngle, glm::vec3 earthRadius, std::shared_ptr<const cs::scene::CelestialObject> earth) {

  std::vector<InfoStruct> structsOut;

  for (int i=0; i < structsIn.size(); i++) {   

    glm::dvec3 p1 = structsIn[i-1].Cartesian; 
    glm::dvec3 p2 = structsIn[i].Cartesian; 
    glm::dvec3 p3 = structsIn[i+1].Cartesian; 

    glm::dvec3 v1 = p2-p1;
    glm::dvec3 v2 = p3-p2;

    if ( i==0 || i== structsIn.size()-1) {
      structsOut.push_back(structsIn[i]);
    }
    else {
      double angle = calculateAngle(structsOut.back(), structsIn[i], structsIn[i+1]);
      if (angle < thresholdAngle) {

        glm::dvec3 firstChildVec = (0.25)*(0.25)*(v2-v1)+2.0*p1;
        InfoStruct firstChildStruct;
        firstChildStruct.Cartesian = firstChildVec; // without actual height
        firstChildStruct.longLatRadians = cs::utils::convert::cartesianToLngLat(firstChildStruct.Cartesian, earthRadius);
        correctHeight (structsIn[i-1], structsIn[i+1], firstChildStruct, earth);
        firstChildStruct.Cartesian = cs::utils::convert::toCartesian(firstChildStruct.longLatRadians, earthRadius, firstChildStruct.overSurfaceHeight);  // with correct height
        structsOut.push_back(firstChildStruct);

        glm::dvec3 secondChildVec = (0.5)*(0.5)*(v2-v1)+2.0*p1;
        InfoStruct secondChildStruct;
        secondChildStruct.Cartesian = secondChildVec; // without actual height
        secondChildStruct.longLatRadians = cs::utils::convert::cartesianToLngLat(secondChildStruct.Cartesian, earthRadius);
        correctHeight (structsIn[i-1], structsIn[i+1], secondChildStruct, earth);
        secondChildStruct.Cartesian = cs::utils::convert::toCartesian(secondChildStruct.longLatRadians, earthRadius, secondChildStruct.overSurfaceHeight);  // with correct height
        structsOut.push_back(secondChildStruct);

        glm::dvec3 thirdChildVec = (0.75)*(0.75)*(v2-v1)+2.0*p1;
        InfoStruct thirdChildStruct;
        thirdChildStruct.Cartesian = thirdChildVec; // without actual height
        thirdChildStruct.longLatRadians = cs::utils::convert::cartesianToLngLat(thirdChildStruct.Cartesian, earthRadius);
        correctHeight (structsIn[i-1], structsIn[i+1], thirdChildStruct, earth);
        thirdChildStruct.Cartesian = cs::utils::convert::toCartesian(thirdChildStruct.longLatRadians, earthRadius, thirdChildStruct.overSurfaceHeight);  // with correct height
        structsOut.push_back(thirdChildStruct);

      }
      else {
        structsOut.push_back(structsIn[i]);
      }
    }
  }
  return structsOut;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

// the function allows to select a server and save in a vector
// (featureTypes) the name of all the datasets contained by it.
//-------------------------------------------------------------  

void Plugin::setWFSServer(std::string URL) {

  if (URL=="None") {
    mPointRenderer = nullptr; 
    mLineStringRenderer = nullptr;
    mPolygonRenderer = nullptr;
    return;
  }
  
  // tinyXML loading.
  mBaseUrl = URL + "?SERVICE=WFS&VERSION=1.3.0";
  std::string mUrl = mBaseUrl + "&REQUEST=GetCapabilities";

  // build the XML request
  std::stringstream xmlStream;      // string stream where the response will be stored
  curlpp::Easy      request;
  request.setOpt(curlpp::options::Url(mUrl));
  request.setOpt(curlpp::options::WriteStream(&xmlStream));
  request.setOpt(curlpp::options::NoSignal(true));
  request.setOpt(curlpp::options::SslVerifyPeer(false));

  // execute the HTTP request and get the file
  try {
    request.perform();
  } catch (std::exception const& e) {
    std::stringstream message;
    message << "WFS capabilities request failed for '" << mUrl << "': '" << e.what() << "'";
    throw std::runtime_error(message.str());
  }

  // transfer the info to a string (doc) and parse it
  std::string docString = xmlStream.str();
  VistaXML::TiXmlDocument doc;
  doc.Parse(docString.c_str());
  if (doc.Error()) {
    std::stringstream message;
    message << "Parsing WFS capabilities failed for '" << mUrl << "': '" << doc.ErrorDesc() << "'";
    throw std::runtime_error(message.str());
  }

  // vector whose components will be all the different featureTypeName
  std::vector<const char*> featureTypes; 

  // save the selected content (in this case, names) of the XML GetCapabilities file
  std::stringstream ss;
  VistaXML::TiXmlElement* featureTypeElement = doc.RootElement() -> FirstChildElement("FeatureTypeList") -> FirstChildElement("FeatureType");
  while  (featureTypeElement) {
    const char* featureTypeName = featureTypeElement -> FirstChildElement ("Name") -> GetText();
    ss << featureTypeName << std::endl;
    featureTypes.push_back(featureTypeName);   

    featureTypeElement = featureTypeElement -> NextSiblingElement ("FeatureType");
  } 

  // clear the GUI list
  mGuiManager->getGui()->callJavascript(
  "CosmoScout.gui.clearDropdown", "wfsOverlays.setWFSFeatureType");
  
  // set the GUI list
  mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wfsOverlays.setWFSFeatureType", "None", "None", false);

  for (int i=0; i < featureTypes.size(); i++) {
    mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wfsOverlays.setWFSFeatureType", featureTypes[i], featureTypes[i], false);
  }
  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// once we select the desired dataset, its information is stored in a class member 
// WFSFeatureCollection-type struct (featureLocation). Additionally, its properies are 
// saved in a DescribeFeatureType-type struct (propertiesStruct), also a class member. 
//-----------------------------------------------------------------------------

void Plugin::setWFSFeatureType(std::string featureType) { 

  mPointRenderer = nullptr; 
  mLineStringRenderer = nullptr;
  mPolygonRenderer = nullptr;

  if (featureType == "None") {
    return;
  }

  std::string mFeatureUrl = mBaseUrl + "&outputFormat=json&REQUEST=GetFeature" + "&typeName=" + featureType;
  logger().info(mFeatureUrl);

  // build the JSON request
  std::stringstream jsonStream;      // Where the response will be stored
  curlpp::Easy      jsonRequest;
  jsonRequest.setOpt(curlpp::options::Url(mFeatureUrl));
  jsonRequest.setOpt(curlpp::options::WriteStream(&jsonStream));
  jsonRequest.setOpt(curlpp::options::NoSignal(true));
  jsonRequest.setOpt(curlpp::options::SslVerifyPeer(false));

  // execute the HTTP request and get the file
  try {
    jsonRequest.perform();            
  } catch (std::exception const& e) {
    std::stringstream message;
    message << "WFS capabilities request failed for '" << mFeatureUrl << "': '" << e.what() << "'";
    throw std::runtime_error(message.str());
  }  

  // transfer the info to a string (doc) and parse it 
  std::string docString = jsonStream.str();
  nlohmann::json data = nlohmann::json::parse(docString);

  // store all the info for a single getCapabilities listed feature
  from_json(data, featureLocation);

  // same thing but for the DescribeFeatureType (DFT)
  //-------------------------------------------------

  std::string mFeaturePropertiesUrl = mBaseUrl + "&outputFormat=application/json&REQUEST=DescribeFeatureType" + "&typeName=" + featureType;

  // build the request
  std::stringstream propertiesStream;      // where the response will be stored
  curlpp::Easy      propertiesRequest;
  jsonRequest.setOpt(curlpp::options::Url(mFeaturePropertiesUrl));
  jsonRequest.setOpt(curlpp::options::WriteStream(&propertiesStream));
  jsonRequest.setOpt(curlpp::options::NoSignal(true));
  jsonRequest.setOpt(curlpp::options::SslVerifyPeer(false));

  // execute the HTTP request and get the file
  try {
    jsonRequest.perform();            
  } catch (std::exception const& e) {
    std::stringstream message;
    message << "WFS capabilities request failed for '" << mFeaturePropertiesUrl << "': '" << e.what() << "'";
    throw std::runtime_error(message.str());
  }  

  // transfer the info to a string (docPropertiesString) and parse it 
  std::string docPropertiesString = propertiesStream.str();
  nlohmann::json propertiesData = nlohmann::json::parse(docPropertiesString);

  // store all the properties of a single getCapabilities listed feature
  from_json(propertiesData, propertiesStruct);

  mGuiManager->getGui()->callJavascript("CosmoScout.wfsOverlays.setFeatureProperties", docPropertiesString);

  

  
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  // here, we loop through the data filled structs we just saved
  // and handle its rendering depending on the type of its geometry
  //---------------------------------------------------------------

  void Plugin::setRendering(double pointSize = 5.0, double lineWidth = 3.0) {   // TODO: double pointSize
    
    logger().info("setRendering::Number of elements: {}", featureLocation.features.size());
    
    auto earth = mSolarSystem->getObject("Earth");
    glm::dvec3 earthRadius = earth->getRadii(); 

    int numPoints = 0;
    int numMultiPoints = 0;
    int numLineStrings = 0;
    int numMultiLineStrings = 0;
    int numPolygons = 0;
    int numMultiPolygons = 0; 
    int numNull = 0;

    std::vector <glm::vec3> colors;

    // just to get the time of processing
    auto startTime = std::chrono::high_resolution_clock::now();

    // as points and multipoints will not use multi-threading
    std::vector <glm::dvec3> pointCoordinates;

    // for the rest of types, with multi-threading 
    std::vector<std::vector<glm::dvec3>> lineStringIntermediateVector;
    lineStringIntermediateVector.reserve(featureLocation.features.size());
    std::vector<std::vector<glm::dvec3>> multiLineStringIntermediateVector;
    multiLineStringIntermediateVector.reserve(featureLocation.features.size());
    std::vector<std::vector<glm::dvec3>> polygonIntermediateVector;
    polygonIntermediateVector.reserve(featureLocation.features.size());
    std::vector<std::vector<glm::dvec3>> multiPolygonIntermediateVector;
    multiPolygonIntermediateVector.reserve(featureLocation.features.size());

    // start the threadPool
    cs::utils::ThreadPool threadPool(std::thread::hardware_concurrency());

    // more than 3 component color flag
    int threeComp = 0;

    for (int l = 0; l < featureLocation.features.size(); l++) {

      // set the color selected by the user via the GUI 
      //-----------------------------------------------   
      glm::dvec3 featureColor = {1.0,1.0,1.0};
      nlohmann::json jsonColor = featureLocation.features[l].properties[mColor];
      if (jsonColor.type() == nlohmann::json::value_t::string) { 
        // try {
          // split the whole string into three substrings
          std::string jsonString = jsonColor.get<std::string>();
          std::vector<std::string> subStrings;
          std::istringstream stream(jsonString);
          std::string subString;
          while (stream >> subString) {
            subStrings.push_back(subString);
          }
          if (subStrings.size() != 3) {
            threeComp++;
            // logger().warn("The color does not have exactly 3 components.");
          }
          // switch from strings to doubles and save them as dvec3 components
          double r = static_cast<double>(std::stoi(subStrings[0]) / 255.0);
          double g = static_cast<double>(std::stoi(subStrings[1]) / 255.0);
          double b = static_cast<double>(std::stoi(subStrings[2]) / 255.0);
          featureColor = {r,g,b};
        } //catch (const std::exception& e) {
          // std::cerr << "error when accessing the JSON: " << e.what() <<std::endl;
      // NOTE THAT IF I WENT BACK TO THE TRY-CATCH METHOD I WOULD HAVE TO ADD SOME } } AFTER THIS PARAGRAPH

      // checking "null" geometry (e.g. DWD -> dwd:Autowarn_Vorhersage)
      if (featureLocation.features[l].geometry == nullptr) {
        numNull++;
        continue;
      }

      std::string type = featureLocation.features[l].geometry->mType;

      if (type == "Point") {
        
          numPoints++;     // quick note here: i++ is not the same as ++i
          std::shared_ptr<Point> point = std::dynamic_pointer_cast<Point>(featureLocation.features[l].geometry); 
          InfoStruct pointStruct;

          pointStruct.longLatDegrees = {point->mCoordinates[0], point->mCoordinates[1]};
          pointStruct.longLatRadians = {cs::utils::convert::toRadians(point->mCoordinates[0]),cs::utils::convert::toRadians(point->mCoordinates[1])};
          if (point->mCoordinates.size() > 2) {
            pointStruct.overSurfaceHeight = point->mCoordinates[2];
            pointStruct.heightComesFromJson = true;
          }
          else {                        
            pointStruct.overSurfaceHeight = earth->getSurface()->getHeight({pointStruct.longLatRadians[0], pointStruct.longLatRadians[1]}) + 10;
            pointStruct.heightComesFromJson = false;
          } 
          pointStruct.Cartesian = cs::utils::convert::toCartesian({pointStruct.longLatRadians[0], pointStruct.longLatRadians[1]}, earthRadius, pointStruct.overSurfaceHeight);
          pointCoordinates.push_back(pointStruct.Cartesian);
          pointCoordinates.push_back(featureColor);
        }

      else if (type == "MultiPoint") {

        numMultiPoints++;
        std::shared_ptr<MultiPoint> multiPoint = std::dynamic_pointer_cast<MultiPoint>(featureLocation.features[l].geometry); 
        InfoStruct multiPointStruct;

        for (int i=0; i < multiPoint->mCoordinates.size(); i++) {
          multiPointStruct.longLatDegrees = {multiPoint->mCoordinates[i][0], multiPoint->mCoordinates[i][1]};
          multiPointStruct.longLatRadians = {cs::utils::convert::toRadians(multiPoint->mCoordinates[i][0]), cs::utils::convert::toRadians(multiPoint->mCoordinates[i][1])};
          
          if (multiPoint->mCoordinates[i].size() > 2) {
            multiPointStruct.overSurfaceHeight = multiPoint->mCoordinates[i][2];
            multiPointStruct.heightComesFromJson = true;
          }
          else {
            multiPointStruct.overSurfaceHeight = earth->getSurface()->getHeight({multiPointStruct.longLatRadians[0], multiPointStruct.longLatRadians[1]}) + 10;
            multiPointStruct.heightComesFromJson = false;
          } 
          multiPointStruct.Cartesian = cs::utils::convert::toCartesian({multiPointStruct.longLatRadians[0], multiPointStruct.longLatRadians[1]}, earthRadius, multiPointStruct.overSurfaceHeight); 
          pointCoordinates.push_back(multiPointStruct.Cartesian);
          pointCoordinates.push_back(featureColor);
        }  
      } 

      else if (type == "LineString") {

        int nIteration = numLineStrings++;
        lineStringIntermediateVector.push_back({});

        auto lineStringProcessing = [&, nIteration, l, featureColor] () {

          std::shared_ptr<LineString> lineString = std::dynamic_pointer_cast<LineString>(featureLocation.features[l].geometry); 
          std::vector<InfoStruct> lineStringAux;
          std::vector<InfoStruct> lineStringStructs;

          for (int i=0; i < lineString->mCoordinates.size(); i++) {
            InfoStruct temporaryStruct;
            temporaryStruct.longLatDegrees = {lineString->mCoordinates[i][0], lineString->mCoordinates[i][1]};
            temporaryStruct.longLatRadians = cs::utils::convert::toRadians(temporaryStruct.longLatDegrees);

            if (lineString->mCoordinates[i].size() > 2) {
              temporaryStruct.overSurfaceHeight = lineString->mCoordinates[i][2];
              temporaryStruct.heightComesFromJson = true;
            }
            else {
              temporaryStruct.overSurfaceHeight = earth->getSurface()->getHeight(temporaryStruct.longLatRadians) + 10;
              temporaryStruct.heightComesFromJson = false;
            }
            temporaryStruct.Cartesian = cs::utils::convert::toCartesian(temporaryStruct.longLatRadians, earthRadius, temporaryStruct.overSurfaceHeight);

            lineStringAux.push_back(temporaryStruct); // lsAux = {v0, v1, ... , vn-1}
          }
          std::vector<InfoStruct> lineStringInterpolated;
          lineStringInterpolated = Interpolation (lineStringAux, 60.0, earthRadius, earth); // lsAux = {v0, interp, v2, interp, v4, ... , vn-1}

          // TODO: lineStringAux = Interpolation (lineStringAux, 60.0, earthRadius, earth)

          for (int i=0; i < lineStringInterpolated.size(); i++) {
            lineStringStructs.push_back(lineStringInterpolated[i]);
            if (i != 0 && i != lineStringInterpolated.size()-1) {     
              lineStringStructs.push_back(lineStringInterpolated[i]);    // TODO: here it would be STRUCT con {v0, 2x interp, v4 , 2x interp, ... , vn-1} 
            } 
          }
          std::vector<glm::dvec3> lineStringVec = generateMidPoint(lineStringStructs, 100000.0, earthRadius, earth, featureColor);
          lineStringIntermediateVector[nIteration].insert(lineStringIntermediateVector[nIteration].end(), lineStringVec.begin(), lineStringVec.end());
          
        };

        threadPool.enqueue(lineStringProcessing); // assign each tread what to do
      }
      
      else if (type == "MultiLineString") {

        int nIteration = numMultiLineStrings++; 
        multiLineStringIntermediateVector.push_back({});

        auto multiLineStringProcessing = [&, nIteration, l] () {

          std::shared_ptr<MultiLineString> multiLineString = std::dynamic_pointer_cast<MultiLineString>(featureLocation.features[l].geometry); 
          
          for (int i=0; i < multiLineString->mCoordinates.size(); i++) { 
            std::vector<InfoStruct> multiLineStringStructs;
            for (int j=0; j < multiLineString->mCoordinates[i].size(); j++) { 
              InfoStruct temporaryStruct;
              temporaryStruct.longLatDegrees = {multiLineString->mCoordinates[i][j][0], multiLineString->mCoordinates[i][j][1]};
              temporaryStruct.longLatRadians = cs::utils::convert::toRadians(temporaryStruct.longLatDegrees);
              
              if (multiLineString->mCoordinates[i][j].size() > 2) {
                temporaryStruct.overSurfaceHeight = multiLineString->mCoordinates[i][j][2];
                temporaryStruct.heightComesFromJson = true;
              }
              else {
                temporaryStruct.overSurfaceHeight = earth->getSurface()->getHeight(temporaryStruct.longLatRadians) + 10;
                temporaryStruct.heightComesFromJson = false;
              }
              temporaryStruct.Cartesian = cs::utils::convert::toCartesian(temporaryStruct.longLatRadians, earthRadius, temporaryStruct.overSurfaceHeight);   
              
              multiLineStringStructs.push_back(temporaryStruct);
              // TODO: We'll have to move this duplication. The ideal place will be after interpolation and before midPointGeneration.
              if (j != 0 && j != multiLineString->mCoordinates[i].size()-1) {          
                multiLineStringStructs.push_back(temporaryStruct);
              } 
            } 
            // TODO: Here would be the interpolation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            std::vector<glm::dvec3> multiLineStringVec = generateMidPoint(multiLineStringStructs, 100000.0, earthRadius, earth, featureColor);
            multiLineStringIntermediateVector[nIteration].insert(multiLineStringIntermediateVector[nIteration].end(), multiLineStringVec.begin(), multiLineStringVec.end());
          }
        };

        threadPool.enqueue(multiLineStringProcessing);      // assign each tread what to do
      } 
    
      else if (type == "Polygon") {

        int nIteration = numPolygons++; 
        polygonIntermediateVector.push_back({});

        auto polygonProcessing = [&, nIteration, l] () {

          std::shared_ptr<Polygon> polygon = std::dynamic_pointer_cast<Polygon>(featureLocation.features[l].geometry); 
          
          for (int i=0; i < polygon->mCoordinates.size(); i++) {
            std::vector<InfoStruct> polygonStructs;
            for (int j=0; j < polygon->mCoordinates[i].size(); j++) {
              InfoStruct temporaryStruct;   
              temporaryStruct.longLatDegrees = {polygon->mCoordinates[i][j][0], polygon->mCoordinates[i][j][1]};
              temporaryStruct.longLatRadians = cs::utils::convert::toRadians(temporaryStruct.longLatDegrees);

              if (polygon->mCoordinates[i][j].size() > 2) {
                temporaryStruct.overSurfaceHeight = polygon->mCoordinates[i][j][2];
                temporaryStruct.heightComesFromJson = true;
              }
              else {
                temporaryStruct.overSurfaceHeight = earth->getSurface()->getHeight(temporaryStruct.longLatRadians) + 10;
                temporaryStruct.heightComesFromJson = false;
              }

              temporaryStruct.Cartesian = cs::utils::convert::toCartesian(temporaryStruct.longLatRadians, earthRadius, temporaryStruct.overSurfaceHeight);  
              
              polygonStructs.push_back(temporaryStruct);
              // TODO: We'll have to move this duplication. The ideal place will be after interpolation and before midPointGeneration.
              if (j != 0 && j != polygon->mCoordinates[i].size()-1) {          
                polygonStructs.push_back(temporaryStruct);
              } 
            }
            // TODO: Here would be the interpolation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            std::vector<glm::dvec3> polygonVec = generateMidPoint(polygonStructs, 100000.0, earthRadius, earth, featureColor);
            polygonIntermediateVector[nIteration].insert(polygonIntermediateVector[nIteration].end(), polygonVec.begin(), polygonVec.end());
          } 
        };

        threadPool.enqueue(polygonProcessing);          // assign each tread what to do              
      }

      else if (type == "MultiPolygon") {

        int nIteration = numMultiPolygons++; 
        multiPolygonIntermediateVector.push_back({});

        auto multiPolygonProcessing = [&, nIteration, l] () {

          std::shared_ptr<MultiPolygon> multiPolygon = std::dynamic_pointer_cast<MultiPolygon>(featureLocation.features[l].geometry);
        
          for (int i=0; i < multiPolygon->mCoordinates.size(); i++) { 
            for (int j=0; j < multiPolygon->mCoordinates[i].size(); j++) {
              std::vector<InfoStruct> multiPolygonStructs;
              for (int k=0; k < multiPolygon->mCoordinates[i][j].size(); k++) {

                InfoStruct temporaryStruct;
                temporaryStruct.longLatDegrees = {multiPolygon->mCoordinates[i][j][k][0], multiPolygon->mCoordinates[i][j][k][1]};
                temporaryStruct.longLatRadians = cs::utils::convert::toRadians(temporaryStruct.longLatDegrees);

                if (multiPolygon->mCoordinates[i][j][k].size() > 2) {
                  temporaryStruct.overSurfaceHeight = multiPolygon->mCoordinates[i][j][k][2];
                  temporaryStruct.heightComesFromJson = true;
                }
                else {
                  temporaryStruct.overSurfaceHeight = earth->getSurface()->getHeight(temporaryStruct.longLatRadians) + 10;
                  temporaryStruct.heightComesFromJson = false;
                }

                temporaryStruct.Cartesian = cs::utils::convert::toCartesian(temporaryStruct.longLatRadians, earthRadius, temporaryStruct.overSurfaceHeight);   
                
                multiPolygonStructs.push_back(temporaryStruct);
                // TODO: We'll have to move this duplication. The ideal place will be after interpolation and before midPointGeneration.
                if (k != 0 && k != multiPolygon->mCoordinates[i][j].size()-1) {          
                  multiPolygonStructs.push_back(temporaryStruct);
                } 
              }
              // TODO: Here would be the interpolation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
              std::vector<glm::dvec3> multiPolygonMidPoints = generateMidPoint(multiPolygonStructs, 100000.0, earthRadius, earth, featureColor);
              multiPolygonIntermediateVector[nIteration].insert(multiPolygonIntermediateVector[nIteration].end(), multiPolygonMidPoints.begin(), multiPolygonMidPoints.end()); 
            }
          } 
        };
        threadPool.enqueue(multiPolygonProcessing);         // assign each tread what to do
      }
      
      else { 
        logger().warn(" {} data could not be rendered", type); 
      }
    }

    // just waiting for the threads to finish
    while (!threadPool.hasFinished()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // TODO: iMPROVE THIS ERROR WARNING
    if (threeComp != 0) {
      logger().warn("The color does not have exactly 3 components.");
    }

    // we have to convert from dvec3 to vec3 before rendering
    //-------------------------------------------------------

    // for points and multipoints, without multithread
    std::vector<glm::vec3> pointCoordinatesRendering;
    for (int i=0; i < pointCoordinates.size(); i++) {
        glm::vec3 coord = static_cast<glm::vec3>(pointCoordinates[i]);
        pointCoordinatesRendering.push_back(coord); 
      }

    // for linestrings and multilinestrings, with multithread
    std::vector<glm::vec3> lineStringCoordinatesRendering;
    for (int i=0; i < lineStringIntermediateVector.size(); i++) {
      if (lineStringIntermediateVector[i].empty()) {
        logger().info("empty line component.");
      }
      for (int j=0; j < lineStringIntermediateVector[i].size(); j++) {
        glm::vec3 coord = static_cast<glm::vec3>(lineStringIntermediateVector[i][j]);
        lineStringCoordinatesRendering.push_back(coord);  
      }  
    }
    for (int i=0; i < multiLineStringIntermediateVector.size(); i++) {
      if (multiLineStringIntermediateVector[i].empty()) {
        logger().info("empty multiline component.");
      }
      for (int j=0; j < multiLineStringIntermediateVector[i].size(); j++) {
        glm::vec3 coord = static_cast<glm::vec3>(multiLineStringIntermediateVector[i][j]);
        lineStringCoordinatesRendering.push_back(coord);
      }  
    }

    // for polygons and multipolygons, with multithread
    std::vector<glm::vec3> polygonCoordinatesRendering;
    for (int i=0; i < polygonIntermediateVector.size(); i++) {
      if (polygonIntermediateVector[i].empty()) {
        logger().info("empty polygon component.");
      }
      for (int j=0; j < polygonIntermediateVector[i].size(); j++) {
        glm::vec3 coord = static_cast<glm::vec3>(polygonIntermediateVector[i][j]);
        polygonCoordinatesRendering.push_back(coord);
      }   
    }
    for (int i=0; i < multiPolygonIntermediateVector.size(); i++) {
      if (multiPolygonIntermediateVector[i].empty()) {
        logger().info("empty multipolygon component.");
      }
      for (int j=0; j < multiPolygonIntermediateVector[i].size(); j++) {
        glm::vec3 coord = static_cast<glm::vec3>(multiPolygonIntermediateVector[i][j]);
        polygonCoordinatesRendering.push_back(coord);
      }   
    }

    if (numNull != 0) {
      logger().info("Number of null elements: {}", numNull);
    }

    // rendering
    //----------

    mPointSize = pointSize;
    mLineWidth = lineWidth;

    
    

    if (!pointCoordinatesRendering.empty()) {
      logger().info( "points: {}, multiPoints: {}. (containing {} points). ", numPoints, numMultiPoints, pointCoordinatesRendering.size()/2 );
      mPointRenderer = std::make_unique<PointRenderer>("Point", pointCoordinatesRendering, mSolarSystem, mAllSettings, mPointSize);
      logger().info("setRendering::PointSize {}", mPointSize);
    }

    if (!lineStringCoordinatesRendering.empty()) {
      logger().info( "lines: {}, multiLines: {}. (containing {} points).", numLineStrings, numMultiLineStrings, (lineStringCoordinatesRendering.size()/2+1)/2 );
      mLineStringRenderer = std::make_unique<FeatureRenderer> ("LineString", lineStringCoordinatesRendering, mSolarSystem, mAllSettings, mLineWidth);
      logger().info("setRendering::LineWidth {}", mLineWidth);
    }

    if (!polygonCoordinatesRendering.empty()) {
      logger().info( "polygons: {}, multiPolygons: {}. (containing {} points).", numPolygons, numMultiPolygons, (polygonCoordinatesRendering.size()/2+1)/2 );
      mPolygonRenderer = std::make_unique<FeatureRenderer> ("Polygon", polygonCoordinatesRendering, mSolarSystem, mAllSettings, mLineWidth);
      logger().info("setRendering::LineWidth {}", mLineWidth);
    }

    if(!mPointRenderer && !mLineStringRenderer && !mPolygonRenderer) {
      logger().info("Server response: {}", jsonStream.str());
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    // printing the time that it took 
    std::chrono::duration<double> diff = endTime - startTime;
    logger().info("Time of execution was: {} s. ", diff.count()); 

    // logger().info("size: {}", pointSize);
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  // Setting a SideBar for the WFS Overlays Plugin
  //----------------------------------------------

  mGuiManager->addSettingsSectionToSideBarFromHTML("WFS Overlays", 
                                                  "device_hub", 
                                                  "../share/resources/gui/wfs_overlays_settings.html");

  mGuiManager->addPluginTabToSideBarFromHTML("WFS Overlays", 
                                            "device_hub", 
                                            "../share/resources/gui/wfs_overlays_tab.html");

  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-wfs-overlays.js");

  // "Enable" callback via GUI
  //--------------------------

  mGuiManager->getGui()->registerCallback("wfsOverlays.setEnabled",
                                          "Enables or disables wfs overlays.",
                                          std::function([this](bool value) { mPluginSettings->mEnabled = value; }));

  mPluginSettings->mEnabled.connectAndTouch([this](bool enable) { mGuiManager->setCheckboxValue("wfsOverlays.setEnabled", enable); });

  // "Set Server" callback via GUI
  //------------------------------

  mGuiManager->getGui()->registerCallback("wfsOverlays.setServer",
                                          "Set the current server to the one with the given name.",
                                          std::function([this](std::string&& name) {
                                            setWFSServer(name);
                                          }));
  mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wfsOverlays.setServer", "None", "None", false);

  // "Set Feature Type" callback via GUI
  //------------------------------------
  mGuiManager->getGui()->registerCallback("wfsOverlays.setWFSFeatureType",
                                          "Set the current feature among all those provided by the server we chose.",
                                          std::function([this](std::string&& name) {
                                            logger().info("-----------------------------------------");
                                            logger().info("Selected new feature: {}", name);
                                            mSelectedFeature = name;
                                            setWFSFeatureType(name);
                                            setRendering();
                                          }));

  mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue", 
                                        "wfsOverlays.setWFSFeatureType", 
                                        "None", "None", false);

  // "Select Color" callback via GUI
  //--------------------------------
  mGuiManager->getGui()->registerCallback("wfsOverlays.setColor",
                                          "Use the chosen radio as the color.",
                                          std::function([this](std::string&& name) {
                                            mColor = name;
                                            setRendering();
                                            logger().info("Color: {}", name);
                                          }));

  // TODO: "Select Date" callback via GUI
  //-------------------------------------

  /*
  mGuiManager->getGui()->registerCallback("wfsOverlays.setTime",
                                          "Use the chosen radio as the time.",
                                          std::function([this](std::string&& name) {
                                            logger().info("Time: {}", name);
                                          }));
  */

  // TODO: "Select Size" callback via GUI
  //-------------------------------------

  mGuiManager->getGui()->registerCallback("wfsOverlays.setSize",
                                          "Use the chosen size for points.",
                                          std::function([this](std::string&& userSelectedSize) {
                                            logger().info("done");
                                            mPointSize = std::stod(userSelectedSize);
                                            setRendering(mPointSize, 3.0);
                                            // logger().info("Size: {}", mPointSize);
                                          }));

  mGuiManager->getGui()->callJavascript("CosmoScout.wfsOverlays.setSize","5.0");

  // TODO: "Select Width" callback via GUI
  //-------------------------------------

  mGuiManager->getGui()->registerCallback("wfsOverlays.setWidth",
                                          "Use the chosen width for lines.",
                                          std::function([this](std::string&& userSelectedWidth) {
                                            logger().info("done");
                                            mLineWidth = std::stod(userSelectedWidth);
                                            setRendering(5.0, mLineWidth);
                                            logger().info("Width: {}", mLineWidth);
                                          }));

  mGuiManager->getGui()->callJavascript("CosmoScout.wfsOverlays.setWidth","3.0");
  
                                          
  onLoad();

  // show the list of servers at the GUI
  //------------------------------------

  std::vector<std::string> serverList = mPluginSettings->mWfs;
  for (int i=0; i < serverList.size(); i++) {
    mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue", 
                                          "wfsOverlays.setServer", 
                                          serverList[i], serverList[i], false);
  }

  logger().info("-----------------------------------------");

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {

  if (mPluginSettings->mEnabled.get()) {
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {

  logger().info("Unloading plugin...");

  // Save settings as this plugin may get reloaded.
  onSave();

  mGuiManager->removeSettingsSection("Wfs Overlays");

  mGuiManager->getGui()->unregisterCallback("wfsOverlays.setEnabled");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-wfs-overlays"), *mPluginSettings);
  }


////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-wfs-overlays"] = *mPluginSettings;
}

} // namespace csp::wfsoverlays

