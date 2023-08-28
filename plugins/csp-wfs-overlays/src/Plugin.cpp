////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "LineRenderer.hpp"
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
    
void from_json(nlohmann::json const& j, Settings& o) {       // here, it is (m)Enabled, (m)Wfs, etc because unlike other classess, the simple_desktop.json file has no constructors
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);         
  cs::core::Settings::deserialize(j, "wfs", o.mWfs);
  cs::core::Settings::deserialize(j, "interpolation", o.mInterpolation); 
}

void to_json(nlohmann::json& j, Settings const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "wfs", o.mWfs);
  cs::core::Settings::serialize(j, "interpolation", o.mInterpolation);
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

void from_json(const nlohmann::json& j, Feature& o) { 
  cs::core::Settings::deserialize(j, "type", o.mType); 
  cs::core::Settings::deserialize(j, "id", o.mId);
  cs::core::Settings::deserialize(j, "geometry", o.mGeometry); 
  cs::core::Settings::deserialize(j, "geometry_name", o.mGeometry_name);
  cs::core::Settings::deserialize(j, "properties", o.mProperties); 
  //cs::core::Settings::deserialize(j, "bbox", o.mBbox); 
} 

void from_json(const nlohmann::json& j, CRS& o) { 
  cs::core::Settings::deserialize(j, "type", o.mType); 
} 

void from_json(const nlohmann::json& j, WFSFeatureCollection& o) { 
  cs::core::Settings::deserialize(j, "type", o.mType); 
  cs::core::Settings::deserialize(j, "features", o.mFeatures); 
  cs::core::Settings::deserialize(j, "totalFeatures", o.mTotalFeatures);
  cs::core::Settings::deserialize(j, "numberMatched", o.mNumberMatched);
  cs::core::Settings::deserialize(j, "numberReturned", o.mNumberReturned); 
  cs::core::Settings::deserialize(j, "timeStamp", o.mTimeStamp);
  cs::core::Settings::deserialize(j, "crs", o.mCrs);
  // cs::core::Settings::deserialize(j, "bbox", o.mBbox);  
} 

////////////////////////////////////////////////////////////////////////////////////////////////////

// now the from_json for the Describe Feature Type
//------------------------------------------------

void from_json(const nlohmann::json& j, Property& o) {
  cs::core::Settings::deserialize(j, "name", o.mName);
  cs::core::Settings::deserialize(j, "maxOccurs", o.mMaxOccurs);
  cs::core::Settings::deserialize(j, "minOccurs", o.mMinOccurs);
  cs::core::Settings::deserialize(j, "nillable", o.mNillable);
  cs::core::Settings::deserialize(j, "type", o.mType);
  cs::core::Settings::deserialize(j, "localType", o.mType);
}

void from_json(const nlohmann::json& j, FeatureType& o) {
  cs::core::Settings::deserialize(j, "typeName", o.mTypeName);
  cs::core::Settings::deserialize(j, "properties", o.mProperties); 
}

void from_json(const nlohmann::json& j, DescribeFeatureType& o) {
  cs::core::Settings::deserialize(j, "elementFormDefault", o.mElementFormDefault);
  cs::core::Settings::deserialize(j, "targetNamespace", o.mTargetNamespace); 
  cs::core::Settings::deserialize(j, "targetPrefix", o.mTargetPrefix);
  cs::core::Settings::deserialize(j, "featureTypes", o.mFeatureTypes); 
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// the function generates midPoints between the Cartesian components of a given vector if its distance is greater than a certain
// threshold. This might be useful to avoid having points rendered under the surface of the Earth. It is remarkable that its input  
// is a vector of structs (structIn) but it returns a vector of doubles (renderingVector). 
//--------------------------------------------------------------------------------------------------------------------------------

std::vector<glm::dvec3> Plugin::generateMidPoint (std::vector <InfoStruct> const& structsIn, float threshold, glm::vec3 earthRadius, std::shared_ptr<const cs::scene::CelestialObject> earth, glm::vec3 featureColor) { 
  
  std::vector <InfoStruct> totalStruct;
  std::vector<glm::dvec3> renderingVector;    // the returned one

  for (int i=0; i < structsIn.size()-1; i++) {  

    // the following definitions will just make the code syntax simpler 
    glm::dvec2 p1LongLatRadians = structsIn[i].mLongLatRadians;
    glm::dvec2 p2LongLatRadians = structsIn[i+1].mLongLatRadians;
    glm::dvec2 p1LongLatDegrees = structsIn[i].mLongLatDegrees;
    glm::dvec2 p2LongLatDegrees = structsIn[i+1].mLongLatDegrees;
    glm::dvec3 p1Cartesian      = structsIn[i].mCartesian;
    glm::dvec3 p2Cartesian      = structsIn[i+1].mCartesian;

    double distance = calculateDistance(structsIn[i], structsIn[i+1], earthRadius);

    if (distance > threshold) { 
      // the structsIn vector already has its inner components duplicated i.e. {v0, v1 (doubled), ... , vn-2 (doubled), vn-1}
      totalStruct.push_back(structsIn[i]);
      int numMidPoints = static_cast<int> (distance/threshold); // number of points we are going to generate between each component
      double segmentLength = distance/threshold;  // distance between those generated points 

      for (int j=1; j <= numMidPoints; j++) {
        InfoStruct temporaryStruct;
        temporaryStruct.mCartesian = glm::mix(p1Cartesian,p2Cartesian,(segmentLength*j)/distance);   // without correct height
        temporaryStruct.mLongLatRadians = cs::utils::convert::cartesianToLngLat(temporaryStruct.mCartesian, earthRadius);
        
        correctHeight (structsIn[i], structsIn[i+1], temporaryStruct, earth);
        
        // correct the cartesian based on height
        temporaryStruct.mCartesian = cs::utils::convert::toCartesian(temporaryStruct.mLongLatRadians, earthRadius, temporaryStruct.mOverSurfaceHeight);  // with correct height
        totalStruct.push_back(temporaryStruct);
        totalStruct.push_back(temporaryStruct);   // now it would look like {v0, ... , vi (doubled), midPoints (doubled)}
      }
    }
    // in case the distance is so small that midPoints are not needed
    else {
      totalStruct.push_back(structsIn[i]);         // in this it would be {v0, ... , vi} (without midPoints)
    } 
  }
  
  totalStruct.push_back(structsIn[structsIn.size() - 1]);   // by doing this we are just adding the last component i.e. {v0, ... , vn-1} 
  
  for (int i=0; i<totalStruct.size(); i++) {    
    renderingVector.push_back(totalStruct[i].mCartesian); // here we add the Cartesian components of all the points and midpoints.   
    renderingVector.push_back(featureColor);  // now we introduce the color
  }

  return renderingVector; // it would be something like {v0.xyz, v0.rgb, .... (duplicated .xyz and .rgb) .... , vn-1.xyz, vn-1.rgb}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// if we just get the xyz components in a spherical coordinate system, it is very likely to have points under the 
// surface of Earth, as it is not a perfect sphere but a body with mountains and slopes. In order to correct that,
// we use the getHeight function provided by cosmoScout. However, sometimes we have a fourth component in our set 
// of coordinates which is the height over the surface of the Earth (specially common in atmospheric datasets like 
// DWD -> dwd:Trajectories). This function will handle both 3 and 4-component data. 
//----------------------------------------------------------------------------------------------------------------

void Plugin::correctHeight (InfoStruct const& struct1, InfoStruct const& struct2, InfoStruct& temporaryStruct, std::shared_ptr<const cs::scene::CelestialObject> earth) {
  
  // for the sake of simplicity
  bool p1Bool = struct1.mHeightComesFromJson;
  bool p2Bool = struct2.mHeightComesFromJson;
  double p1Height = struct1.mOverSurfaceHeight;
  double p2Height = struct2.mOverSurfaceHeight;

  if (p1Bool || p2Bool) {
    temporaryStruct.mOverSurfaceHeight = ((p1Height + p2Height)/2);
    temporaryStruct.mHeightComesFromJson = true;
  } 
  else {
    temporaryStruct.mOverSurfaceHeight = earth->getSurface()->getHeight(temporaryStruct.mLongLatRadians);
    temporaryStruct.mHeightComesFromJson = false;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// the function takes as an input the earthRadius and a couple of cartesian points,
// whose Great Circle distance will be returned (We will use the Haversine method).
//---------------------------------------------------------------------------------

double Plugin::calculateDistance(InfoStruct const& p1, InfoStruct const& p2, glm::vec3 earthRadius) { 
  
  // great circle distance using Haversine formula     
  double rootArgument = cos(p1.mLongLatRadians[1]) * cos(p2.mLongLatRadians[1]) * cos(p1.mLongLatRadians[0] - p2.mLongLatRadians[0]) + sin(p1.mLongLatRadians[1]) * sin(p2.mLongLatRadians[1]);
  double averageEarthRadius = (earthRadius[0] + earthRadius[1] + earthRadius[2])/3;
  double distanceHaversine = (averageEarthRadius + (p1.mOverSurfaceHeight+p2.mOverSurfaceHeight)/2)  * acos(rootArgument);

  // staight-line distance
  double distanceStraight = glm::distance(p1.mCartesian,p2.mCartesian);
  
  // comparison
  if (distanceStraight>distanceHaversine) {
    return distanceStraight;
  }
  else {
    return distanceHaversine;    
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// the aim of the function is simply returning the angle (in degrees) between
// three points. These points will be introduced as components of structs. 
//---------------------------------------------------------------------------

double Plugin::calculateAngle (InfoStruct const& previousPoint, InfoStruct const& middlePoint, InfoStruct const& nextPoint) {

  glm::dvec3 vec1 = middlePoint.mCartesian - previousPoint.mCartesian;
  glm::dvec3 vec2 = nextPoint.mCartesian - middlePoint.mCartesian;

  double dotProduct = glm::dot(glm::normalize(vec1),glm::normalize(vec2));
  double angleRad =  glm::acos(dotProduct); // in radians
  double PI = glm::pi<double>();
  double angleDeg = angleRad * (180.0 / PI);
  return angleDeg;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// the function checks if a certain line/polygon has sharp angles. If that's the case, it generates 3 midPoints in order to get a smoother 
// rendering. Again, the threshold angle is up to user. The theoretical base behind this construction is the Bézier-curves generation. 
//----------------------------------------------------------------------------------------------------------------------------------------

std::vector<InfoStruct> Plugin::interpolation (std::vector<InfoStruct> const& structsIn, double thresholdAngle, glm::vec3 earthRadius, std::shared_ptr<const cs::scene::CelestialObject> earth) {

  std::vector<InfoStruct> structsOut;

  for (int i=0; i < structsIn.size(); i++) {   

    if ( i==0 || i== structsIn.size()-1) {
      structsOut.push_back(structsIn[i]);
      continue;
    }

    glm::dvec3 p1 = structsOut.back().mCartesian; 
    glm::dvec3 p2 = structsIn[i].mCartesian; 
    glm::dvec3 p3 = structsIn[i+1].mCartesian; 

    glm::dvec3 v1 = p2-p1;
    glm::dvec3 v2 = p3-p2;

    double angle = calculateAngle(structsOut.back(), structsIn[i], structsIn[i+1]);

    if (angle < thresholdAngle) {
      
      // first midPoint. It has a 0.25 interpolation coefficient.
      glm::dvec3 firstChildVec =  p1+(0.25)*v1+(0.25)*p2+(0.25)*(0.25)*v2-p1*(0.25)-(0.25)*(0.25)*v1;
      InfoStruct firstChildStruct;
      firstChildStruct.mCartesian = firstChildVec; // without actual height
      firstChildStruct.mLongLatRadians = cs::utils::convert::cartesianToLngLat(firstChildStruct.mCartesian, earthRadius);
      correctHeight (structsIn[i-1], structsIn[i+1], firstChildStruct, earth);
      firstChildStruct.mCartesian = cs::utils::convert::toCartesian(firstChildStruct.mLongLatRadians, earthRadius, firstChildStruct.mOverSurfaceHeight);  // with correct height
      structsOut.push_back(firstChildStruct);

      // second midPoint. It has a 0.5 interpolation coefficient.
      glm::dvec3 secondChildVec = p1+(0.5)*v1+(0.5)*p2+(0.5)*(0.5)*v2-p1*(0.5)-(0.5)*(0.5)*v1;
      InfoStruct secondChildStruct;
      secondChildStruct.mCartesian = secondChildVec; // without actual height
      secondChildStruct.mLongLatRadians = cs::utils::convert::cartesianToLngLat(secondChildStruct.mCartesian, earthRadius);
      correctHeight (structsIn[i-1], structsIn[i+1], secondChildStruct, earth);
      secondChildStruct.mCartesian = cs::utils::convert::toCartesian(secondChildStruct.mLongLatRadians, earthRadius, secondChildStruct.mOverSurfaceHeight);  // with correct height
      structsOut.push_back(secondChildStruct);

      // third midPoint. It has a 0.75 interpolation coefficient.
      glm::dvec3 thirdChildVec = p1+(0.75)*v1+(0.75)*p2+(0.75)*(0.75)*v2-p1*(0.75)-(0.75)*(0.75)*v1;
      InfoStruct thirdChildStruct;
      thirdChildStruct.mCartesian = thirdChildVec; // without actual height
      thirdChildStruct.mLongLatRadians = cs::utils::convert::cartesianToLngLat(thirdChildStruct.mCartesian, earthRadius);
      correctHeight (structsIn[i-1], structsIn[i+1], thirdChildStruct, earth);
      thirdChildStruct.mCartesian = cs::utils::convert::toCartesian(thirdChildStruct.mLongLatRadians, earthRadius, thirdChildStruct.mOverSurfaceHeight);  // with correct height
      structsOut.push_back(thirdChildStruct);
    }
    else {
      structsOut.push_back(structsIn[i]);
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
  std::string url = mBaseUrl + "&REQUEST=GetCapabilities";

  // build the XML request
  std::stringstream xmlStream;  // where the response will be stored
  curlpp::Easy      request;
  request.setOpt(curlpp::options::Url(url));
  request.setOpt(curlpp::options::WriteStream(&xmlStream));
  request.setOpt(curlpp::options::NoSignal(true));
  request.setOpt(curlpp::options::SslVerifyPeer(false));

  // execute the HTTP request and get the file
  try {
    request.perform();
  } catch (std::exception const& e) {
    std::stringstream message;
    message << "WFS capabilities request failed for '" << url << "': '" << e.what() << "'";
    throw std::runtime_error(message.str());
  }

  // transfer the info to a string (doc) and parse it
  std::string docString = xmlStream.str();
  VistaXML::TiXmlDocument doc;
  doc.Parse(docString.c_str());
  if (doc.Error()) {
    std::stringstream message;
    message << "Parsing WFS capabilities failed for '" << url << "': '" << doc.ErrorDesc() << "'";
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
// WFSFeatureCollection-type struct (mFeatureLocation). Additionally, its properies are 
// saved in a DescribeFeatureType-type struct (mPropertiesStruct), also a class member. 
//-------------------------------------------------------------------------------------

void Plugin::setWFSFeatureType(std::string featureType) { 

  mPointRenderer = nullptr; 
  mLineStringRenderer = nullptr;
  mPolygonRenderer = nullptr;

  if (featureType=="None") { 
    return;
  }

  std::string featureUrl = mBaseUrl + "&outputFormat=json&REQUEST=GetFeature" + "&typeName=" + featureType;
  logger().info(featureUrl);

  // build the JSON request
  std::stringstream mJsonStream;      // Where the response will be stored
  curlpp::Easy      jsonRequest;
  jsonRequest.setOpt(curlpp::options::Url(featureUrl));
  jsonRequest.setOpt(curlpp::options::WriteStream(&mJsonStream));
  jsonRequest.setOpt(curlpp::options::NoSignal(true));
  jsonRequest.setOpt(curlpp::options::SslVerifyPeer(false));

  // execute the HTTP request and get the file
  try {
    jsonRequest.perform();            
  } catch (std::exception const& e) {
    std::stringstream message;
    message << "WFS capabilities request failed for '" << featureUrl << "': '" << e.what() << "'";
    throw std::runtime_error(message.str());
  }  

  // transfer the info to a string (doc) and parse it 
  std::string docString = mJsonStream.str();
  nlohmann::json data = nlohmann::json::parse(docString);

  // store all the info for a single getCapabilities listed feature
  from_json(data, mFeatureLocation);

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
  from_json(propertiesData, mPropertiesStruct);

  mGuiManager->getGui()->callJavascript("CosmoScout.wfsOverlays.setFeatureProperties", docPropertiesString);
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  // here, we loop through the data filled structs we just saved and 
  // handle its rendering depending on the type of its geometry
  //----------------------------------------------------------------

  void Plugin::setRendering(double pointSize = 0.02, double lineWidth = 2.0) {   
    
    logger().info("-------");
    logger().info("setRendering::Number of elements: {}", mFeatureLocation.mFeatures.size());
    
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
    lineStringIntermediateVector.reserve(mFeatureLocation.mFeatures.size());
    std::vector<std::vector<glm::dvec3>> multiLineStringIntermediateVector;
    multiLineStringIntermediateVector.reserve(mFeatureLocation.mFeatures.size());
    std::vector<std::vector<glm::dvec3>> polygonIntermediateVector;
    polygonIntermediateVector.reserve(mFeatureLocation.mFeatures.size());
    std::vector<std::vector<glm::dvec3>> multiPolygonIntermediateVector;
    multiPolygonIntermediateVector.reserve(mFeatureLocation.mFeatures.size());

    // start the threadPool
    cs::utils::ThreadPool threadPool(std::thread::hardware_concurrency());
    // in case of crashing, we can always try using one thread only: cs::utils::ThreadPool threadPool(1);

    // more than 3 component color flag
    int threeComp = 0;

    for (int l = 0; l < mFeatureLocation.mFeatures.size(); l++) {

      // set the color selected by the user via the GUI 
      //-----------------------------------------------   
      glm::dvec3 featureColor = {1.0,1.0,1.0};
      nlohmann::json jsonColor = mFeatureLocation.mFeatures[l].mProperties[mColor];
      if (jsonColor.type() == nlohmann::json::value_t::string) { 
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
          }
          // switch from strings to doubles and save them as dvec3 components
          double r = static_cast<double>(std::stoi(subStrings[0]) / 255.0);
          double g = static_cast<double>(std::stoi(subStrings[1]) / 255.0);
          double b = static_cast<double>(std::stoi(subStrings[2]) / 255.0);
          featureColor = {r,g,b};
        }
        
      // checking "null" geometry (e.g. DWD -> dwd:Autowarn_Vorhersage)
      if (mFeatureLocation.mFeatures[l].mGeometry == nullptr) {
        numNull++;
        continue;
      }

      std::string type = mFeatureLocation.mFeatures[l].mGeometry->mType;

      if (type == "Point") {
        
          numPoints++;     
          std::shared_ptr<Point> point = std::dynamic_pointer_cast<Point>(mFeatureLocation.mFeatures[l].mGeometry); 
          InfoStruct pointStruct;

          pointStruct.mLongLatDegrees = {point->mCoordinates[0], point->mCoordinates[1]};
          pointStruct.mLongLatRadians = {cs::utils::convert::toRadians(point->mCoordinates[0]),cs::utils::convert::toRadians(point->mCoordinates[1])};
          // if there is a 4th component for height over the surface
          if (point->mCoordinates.size() > 2) {
            pointStruct.mOverSurfaceHeight = point->mCoordinates[2];
            pointStruct.mHeightComesFromJson = true;
          }
          else {  // if there is not a 4th component                     
            pointStruct.mOverSurfaceHeight = earth->getSurface()->getHeight({pointStruct.mLongLatRadians[0], pointStruct.mLongLatRadians[1]}) + 10;
            pointStruct.mHeightComesFromJson = false;
          } 
          pointStruct.mCartesian = cs::utils::convert::toCartesian({pointStruct.mLongLatRadians[0], pointStruct.mLongLatRadians[1]}, earthRadius, pointStruct.mOverSurfaceHeight);
          pointCoordinates.push_back(pointStruct.mCartesian);
          pointCoordinates.push_back(featureColor);
        }

      else if (type == "MultiPoint") {

        numMultiPoints++;
        std::shared_ptr<MultiPoint> multiPoint = std::dynamic_pointer_cast<MultiPoint>(mFeatureLocation.mFeatures[l].mGeometry); 
        InfoStruct multiPointStruct;

        for (int i=0; i < multiPoint->mCoordinates.size(); i++) {
          multiPointStruct.mLongLatDegrees = {multiPoint->mCoordinates[i][0], multiPoint->mCoordinates[i][1]};
          multiPointStruct.mLongLatRadians = {cs::utils::convert::toRadians(multiPoint->mCoordinates[i][0]), cs::utils::convert::toRadians(multiPoint->mCoordinates[i][1])};
          // if there is a 4th component for height over the surface
          if (multiPoint->mCoordinates[i].size() > 2) {
            multiPointStruct.mOverSurfaceHeight = multiPoint->mCoordinates[i][2];
            multiPointStruct.mHeightComesFromJson = true;
          }
          else {  // if there is not a 4th component
            multiPointStruct.mOverSurfaceHeight = earth->getSurface()->getHeight({multiPointStruct.mLongLatRadians[0], multiPointStruct.mLongLatRadians[1]}) + 10;
            multiPointStruct.mHeightComesFromJson = false;
          } 
          multiPointStruct.mCartesian = cs::utils::convert::toCartesian({multiPointStruct.mLongLatRadians[0], multiPointStruct.mLongLatRadians[1]}, earthRadius, multiPointStruct.mOverSurfaceHeight); 
          pointCoordinates.push_back(multiPointStruct.mCartesian);
          pointCoordinates.push_back(featureColor);
        }  
      } 

      else if (type == "LineString") {

        int nIteration = numLineStrings++;
        lineStringIntermediateVector.push_back({});

        // we need to use a lambda function to assign it as a task for the threads
        auto lineStringProcessing = [&, nIteration, l] () {

          std::vector<InfoStruct> lineStringAux;
          std::shared_ptr<LineString> lineString = std::dynamic_pointer_cast<LineString>(mFeatureLocation.mFeatures[l].mGeometry); 
          std::vector<InfoStruct> lineStringStructs;

          for (int i=0; i < lineString->mCoordinates.size(); i++) {
            InfoStruct temporaryStruct{};
            temporaryStruct.mLongLatDegrees = {lineString->mCoordinates[i][0], lineString->mCoordinates[i][1]};
            temporaryStruct.mLongLatRadians = cs::utils::convert::toRadians(temporaryStruct.mLongLatDegrees);
            // if there is a 4th component for height over the surface
            if (lineString->mCoordinates[i].size() > 2) {
              temporaryStruct.mOverSurfaceHeight = lineString->mCoordinates[i][2];
              temporaryStruct.mHeightComesFromJson = true;
            }
            else {  // if there is not a 4th component
              temporaryStruct.mOverSurfaceHeight = earth->getSurface()->getHeight(temporaryStruct.mLongLatRadians) + 10;
              temporaryStruct.mHeightComesFromJson = false;
            }
            temporaryStruct.mCartesian = cs::utils::convert::toCartesian(temporaryStruct.mLongLatRadians, earthRadius, temporaryStruct.mOverSurfaceHeight);

            if (mPluginSettings->mInterpolation.get()) {  
              lineStringAux.push_back(temporaryStruct);    
            }
            else {  // duplicating the components when no interpolation is used
              lineStringStructs.push_back(temporaryStruct);
              if (i != 0 && i != lineString->mCoordinates.size()-1) {          
                lineStringStructs.push_back(temporaryStruct);
              }
            } 
          }

          if (mPluginSettings->mInterpolation.get()) {  // in case we want to use the interpolation method
            std::vector<InfoStruct> lineStringInterpolated;
            lineStringInterpolated = interpolation (lineStringAux, 60.0, earthRadius, earth); 
            for (int a=0; a < lineStringInterpolated.size(); a++) { // duplicating the components
              lineStringStructs.push_back(lineStringInterpolated[a]);
              if (a != 0 && a != lineStringInterpolated.size()-1) { 
                lineStringStructs.push_back(lineStringInterpolated[a]);    
              } 
            }
          }
          std::vector<glm::dvec3> lineStringVec = generateMidPoint(lineStringStructs, 100000.0, earthRadius, earth, featureColor);
          lineStringIntermediateVector[nIteration].insert(lineStringIntermediateVector[nIteration].end(), lineStringVec.begin(), lineStringVec.end());
        };
        threadPool.enqueue(lineStringProcessing); // assign each thread what to do
      }
      
      else if (type == "MultiLineString") {

        int nIteration = numMultiLineStrings++; 
        multiLineStringIntermediateVector.push_back({});

        // we need to use a lambda function to assign it as a task for the threads
        auto multiLineStringProcessing = [&, nIteration, l] () {

          std::shared_ptr<MultiLineString> multiLineString = std::dynamic_pointer_cast<MultiLineString>(mFeatureLocation.mFeatures[l].mGeometry); 
          
          for (int i=0; i < multiLineString->mCoordinates.size(); i++) { 
            std::vector<InfoStruct> multiLineStringStructs;
            std::vector<InfoStruct> multiLineStringAux; 
            for (int j=0; j < multiLineString->mCoordinates[i].size(); j++) { 
              InfoStruct temporaryStruct{};
              temporaryStruct.mLongLatDegrees = {multiLineString->mCoordinates[i][j][0], multiLineString->mCoordinates[i][j][1]};
              temporaryStruct.mLongLatRadians = cs::utils::convert::toRadians(temporaryStruct.mLongLatDegrees);
              // if there is a 4th component for height over the surface
              if (multiLineString->mCoordinates[i][j].size() > 2) {
                temporaryStruct.mOverSurfaceHeight = multiLineString->mCoordinates[i][j][2];
                temporaryStruct.mHeightComesFromJson = true;
              }
              else {  // if there is not a 4th component
                temporaryStruct.mOverSurfaceHeight = earth->getSurface()->getHeight(temporaryStruct.mLongLatRadians) + 10;
                temporaryStruct.mHeightComesFromJson = false;
              }
              temporaryStruct.mCartesian = cs::utils::convert::toCartesian(temporaryStruct.mLongLatRadians, earthRadius, temporaryStruct.mOverSurfaceHeight); 

              if (mPluginSettings->mInterpolation.get()) {  
                multiLineStringAux.push_back(temporaryStruct);  
              } 
              else { // duplicating the components when no interpolation is used
                multiLineStringStructs.push_back(temporaryStruct);
                if (j != 0 && j != multiLineString->mCoordinates[i].size()-1) {          
                  multiLineStringStructs.push_back(temporaryStruct);
                } 
              } 
            }
            if (mPluginSettings->mInterpolation.get()) {  // in case we want to use the interpolation method
              std::vector<InfoStruct> multiLineStringInterpolated;
              multiLineStringInterpolated = interpolation (multiLineStringAux, 60.0, earthRadius, earth); 
              for (int a=0; a < multiLineStringInterpolated.size(); a++) {  // duplicating the components
                multiLineStringStructs.push_back(multiLineStringInterpolated[a]);
                if (a != 0 && a != multiLineStringInterpolated.size()-1) { 
                  multiLineStringStructs.push_back(multiLineStringInterpolated[a]);    
                } 
              }
            }
            std::vector<glm::dvec3> multiLineStringVec = generateMidPoint(multiLineStringStructs, 100000.0, earthRadius, earth, featureColor);
            multiLineStringIntermediateVector[nIteration].insert(multiLineStringIntermediateVector[nIteration].end(), multiLineStringVec.begin(), multiLineStringVec.end());
          }
        };
        threadPool.enqueue(multiLineStringProcessing);      // assign each thread what to do
      }

      else if (type == "Polygon") {

        int nIteration = numPolygons++; 
        polygonIntermediateVector.push_back({});
        
        // we need to use a lambda function to assign it as a task for the threads
        auto polygonProcessing = [&, nIteration, l] () {
          std::shared_ptr<Polygon> polygon = std::dynamic_pointer_cast<Polygon>(mFeatureLocation.mFeatures[l].mGeometry); 

          for (int i=0; i < polygon->mCoordinates.size(); i++) {
            
            std::vector<InfoStruct> polygonStructs;
            std::vector<InfoStruct> polygonAux; 

            for (int j=0; j < polygon->mCoordinates[i].size(); j++) {
              InfoStruct temporaryStruct = {};
              temporaryStruct.mLongLatDegrees = {polygon->mCoordinates[i][j][0], polygon->mCoordinates[i][j][1]};
              temporaryStruct.mLongLatRadians = cs::utils::convert::toRadians(temporaryStruct.mLongLatDegrees);
              // if there is a 4th component for height over the surface
              if (polygon->mCoordinates[i][j].size() > 2) {
                temporaryStruct.mOverSurfaceHeight = polygon->mCoordinates[i][j][2];
                temporaryStruct.mHeightComesFromJson = true;
              }
              else {  // if there is not a 4th component
                temporaryStruct.mOverSurfaceHeight = earth->getSurface()->getHeight(temporaryStruct.mLongLatRadians) + 10;
                temporaryStruct.mHeightComesFromJson = false;
              }
              temporaryStruct.mCartesian = cs::utils::convert::toCartesian(temporaryStruct.mLongLatRadians, earthRadius, temporaryStruct.mOverSurfaceHeight);  
              
              if (mPluginSettings->mInterpolation.get()) {  
                polygonAux.push_back(temporaryStruct); 
              } 
              else {  // duplicating the components when no interpolation is used
                polygonStructs.push_back(temporaryStruct);
                if (j != 0 && j != polygon->mCoordinates[i].size()-1) {          
                  polygonStructs.push_back(temporaryStruct);
                } 
              }
            }
            if (mPluginSettings->mInterpolation.get()) {  // in case we want to use the interpolation method
              std::vector<InfoStruct> polygonInterpolated;
              polygonInterpolated = interpolation (polygonAux, 60.0, earthRadius, earth); 
              for (int a=0; a < polygonInterpolated.size(); a++) {  // duplicating the components
                polygonStructs.push_back(polygonInterpolated[a]);
                if (a != 0 && a != polygonInterpolated.size()-1) { 
                  polygonStructs.push_back(polygonInterpolated[a]);   
                } 
              }
            }
            std::vector<glm::dvec3> polygonVec = generateMidPoint(polygonStructs, 100000.0, earthRadius, earth, featureColor);
            polygonIntermediateVector[nIteration].insert(polygonIntermediateVector[nIteration].end(), polygonVec.begin(), polygonVec.end());
          } 
        };
        threadPool.enqueue(polygonProcessing);          // assign each thread what to do              
      }
      
      else if (type == "MultiPolygon") {

        int nIteration = numMultiPolygons++; 
        multiPolygonIntermediateVector.push_back({});

        // we need to use a lambda function to assign it as a task for the threads
        auto multiPolygonProcessing = [&, nIteration, l] () {

          std::shared_ptr<MultiPolygon> multiPolygon = std::dynamic_pointer_cast<MultiPolygon>(mFeatureLocation.mFeatures[l].mGeometry);
        
          for (int i=0; i < multiPolygon->mCoordinates.size(); i++) { 

            for (int j=0; j < multiPolygon->mCoordinates[i].size(); j++) {

              std::vector<InfoStruct> multiPolygonStructs;
              std::vector<InfoStruct> multiPolygonAux; 

              for (int k=0; k < multiPolygon->mCoordinates[i][j].size(); k++) {
                InfoStruct temporaryStruct;
                temporaryStruct.mLongLatDegrees = {multiPolygon->mCoordinates[i][j][k][0], multiPolygon->mCoordinates[i][j][k][1]};
                temporaryStruct.mLongLatRadians = cs::utils::convert::toRadians(temporaryStruct.mLongLatDegrees);
                // if there is a 4th component for height over the surface
                if (multiPolygon->mCoordinates[i][j][k].size() > 2) {
                  temporaryStruct.mOverSurfaceHeight = multiPolygon->mCoordinates[i][j][k][2];
                  temporaryStruct.mHeightComesFromJson = true;
                }
                else {  // if there is not a 4th component
                  temporaryStruct.mOverSurfaceHeight = earth->getSurface()->getHeight(temporaryStruct.mLongLatRadians) + 10;
                  temporaryStruct.mHeightComesFromJson = false;
                }
                temporaryStruct.mCartesian = cs::utils::convert::toCartesian(temporaryStruct.mLongLatRadians, earthRadius, temporaryStruct.mOverSurfaceHeight);   
                
                if (mPluginSettings->mInterpolation.get()) { 
                    multiPolygonAux.push_back(temporaryStruct);  
                } 
                else {  // duplicating the components when no interpolation is used
                  multiPolygonStructs.push_back(temporaryStruct);
                  if (k != 0 && k != multiPolygon->mCoordinates[i][j].size()-1) {          
                    multiPolygonStructs.push_back(temporaryStruct);
                  } 
                }  
              }
              if (mPluginSettings->mInterpolation.get()) {  // in case we want to use the interpolation method
                std::vector<InfoStruct> multiPolygonInterpolated;
                multiPolygonInterpolated = interpolation (multiPolygonAux, 60.0, earthRadius, earth); 
                for (int a=0; a < multiPolygonInterpolated.size(); a++) {
                  multiPolygonStructs.push_back(multiPolygonInterpolated[a]);
                  if (a != 0 && a != multiPolygonInterpolated.size()-1) { 
                    multiPolygonStructs.push_back(multiPolygonInterpolated[a]);    
                  } 
                }
              }
              std::vector<glm::dvec3> multiPolygonMidPoints = generateMidPoint(multiPolygonStructs, 100000.0, earthRadius, earth, featureColor);
              multiPolygonIntermediateVector[nIteration].insert(multiPolygonIntermediateVector[nIteration].end(), multiPolygonMidPoints.begin(), multiPolygonMidPoints.end()); 
            }
          } 
        };
        threadPool.enqueue(multiPolygonProcessing);         // assign each thread what to do
      }

      else { 
        logger().warn(" {} data could not be rendered", type); 
      }
    }

    // just waiting for the threads to finish
    while (!threadPool.hasFinished()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (threeComp != 0) {
      logger().warn("The color does not have exactly 3 components.");
    }

    // by now, we have already saved all the coordinates and colors in 
    // the designated vectorfor every different type. Before rendering
    // with openGL, we need to convert from dvec3 to vec3.
    //----------------------------------------------------------------

    // conversion for points and multipoints, without multithread
    std::vector<glm::vec3> pointCoordinatesRendering;
    for (int i=0; i < pointCoordinates.size(); i++) {
        glm::vec3 coord = static_cast<glm::vec3>(pointCoordinates[i]);
        pointCoordinatesRendering.push_back(coord); 
      }

    // conversion for linestrings and multilinestrings, with multithread
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

    // conversion for polygons and multipolygons, with multithread
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
      mPointRenderer = std::make_unique<PointRenderer>(pointCoordinatesRendering, mSolarSystem, mAllSettings, mPointSize, mPluginSettings);
      logger().info("setRendering::PointSize {}", mPointSize);
    }

    if (!lineStringCoordinatesRendering.empty()) {
      logger().info( "lines: {}, multiLines: {}. (containing {} points).", numLineStrings, numMultiLineStrings, (lineStringCoordinatesRendering.size()/2+1)/2 );
      mLineStringRenderer = std::make_unique<LineRenderer> (lineStringCoordinatesRendering, mSolarSystem, mAllSettings, mLineWidth, mPluginSettings);
      logger().info("setRendering::LineWidth {}", mLineWidth);
    }

    if (!polygonCoordinatesRendering.empty()) {
      logger().info( "polygons: {}, multiPolygons: {}. (containing {} points).", numPolygons, numMultiPolygons, (polygonCoordinatesRendering.size()/2+1)/2 );
      mPolygonRenderer = std::make_unique<LineRenderer> (polygonCoordinatesRendering, mSolarSystem, mAllSettings, mLineWidth, mPluginSettings);
      logger().info("setRendering::LineWidth {}", mLineWidth);
    }

    if(!mPointRenderer && !mLineStringRenderer && !mPolygonRenderer) {
      logger().info("Server response: {}", mJsonStream.str());
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    // registering the time that the vector assignment and rendering took 
    std::chrono::duration<double> diff = endTime - startTime;

    // We can print that time by just uncommenting the following line:
    // logger().info("Time of execution was: {} s. ", diff.count());    
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

  // "Interpolation" callback via GUI
  //---------------------------------

  mGuiManager->getGui()->registerCallback("wfsOverlays.setInterpolation",
                                          "Enables or disables interpolation rendering.",
                                          std::function([this](bool value) { 
                                            mPluginSettings->mInterpolation = value;
                                            logger().info("-------"); 
                                            logger().info("Interpolation: {}", mPluginSettings->mInterpolation.get());
                                            setRendering();
                                          }));

  mPluginSettings->mInterpolation.connectAndTouch([this](bool interpolation) { mGuiManager->setCheckboxValue("wfsOverlays.setInterpolation", interpolation); });

  // "Set Server" callback via GUI
  //------------------------------

  mGuiManager->getGui()->registerCallback("wfsOverlays.setServer",
                                          "Set the current server to the one with the given name.",
                                          std::function([this](std::string&& name) {
                                            logger().info("-----------------------------------------");
                                            logger().info("Selected new server: {}", name);
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
                                            if (!(name == "None")) {
                                              setRendering();
                                            }
                                          }));

  mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue", 
                                        "wfsOverlays.setWFSFeatureType", 
                                        "None", "None", false);

  // "Select Color" callback via GUI (if we wanted to do a "Select Date", it would be similar)
  //------------------------------------------------------------------------------------------
  mGuiManager->getGui()->registerCallback("wfsOverlays.setColor",
                                          "Use the chosen radio as the color.",
                                          std::function([this](std::string&& name) {
                                            mColor = name;
                                            setRendering();
                                            logger().info("-------");
                                            logger().info("setColor:: {}", name);
                                          }));

  // "Select Size" callback via GUI
  //-------------------------------
  mGuiManager->getGui()->registerCallback("wfsOverlays.setSize",
                                          "Use the chosen size for points.",
                                          std::function([this](std::string&& userSelectedSize) {
                                            mPointSize = std::stod(userSelectedSize);
                                            setRendering(mPointSize, 2.0);
                                          }));

  mGuiManager->getGui()->callJavascript("CosmoScout.wfsOverlays.setSize","1.0");

  // "Select Width" callback via GUI
  //--------------------------------
  mGuiManager->getGui()->registerCallback("wfsOverlays.setWidth",
                                          "Use the chosen width for lines.",
                                          std::function([this](std::string&& userSelectedWidth) {
                                            mLineWidth = std::stod(userSelectedWidth);
                                            setRendering(0.02, mLineWidth);
                                          }));

  mGuiManager->getGui()->callJavascript("CosmoScout.wfsOverlays.setWidth","1.0");
  
  onLoad();

  // show the list of servers at the GUI
  //------------------------------------
  std::vector<std::string> serverList = mPluginSettings->mWfs;
  for (int i=0; i < serverList.size(); i++) {
    mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue", 
                                          "wfsOverlays.setServer", 
                                          serverList[i], serverList[i], false);
  }
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

  mGuiManager->getGui()->unregisterCallback("wfsOverlays.setInterpolation");

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

