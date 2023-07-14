////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "FeatureRenderer.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "logger.hpp"

#include <iostream>

#include <VistaTools/tinyXML/tinyxml.h>

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>


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

// binding the components of the Settings struct (from the Plugin class)
// defined at the header with the elements at the simple_desktop.json
    
void from_json(nlohmann::json const& j, Plugin::Settings& o) {       // here, it is (m)Enabled, (m)Wfs because the
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);         // simple_desktop.json file has no constructors
  cs::core::Settings::deserialize(j, "wfs", o.mWfs); 
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "wfs", o.mWfs);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Just the from_json for all the different structs defined at the WFSTypes.hpp

void from_json(const nlohmann::json& j, std::shared_ptr<GeometryBase>& o) {
  
  // the following if is for those datasets containing "null" geometry (e.g. DWD -> dwd:Autowarn_Vorhersage)
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

// Here it would be the Properties1

void from_json(const nlohmann::json& j, Feature& o) { 
  cs::core::Settings::deserialize(j, "type", o.type); 
  cs::core::Settings::deserialize(j, "id", o.id);
  cs::core::Settings::deserialize(j, "geometry", o.geometry); // Struct
  cs::core::Settings::deserialize(j, "geometry_name", o.geometry_name);
  //cs::core::Settings::deserialize(j, "properties", o.properties); // Prop
  //cs::core::Settings::deserialize(j, "bbox", o.bbox); 
} 

// Here it would be the Properties2

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

// the function generates midPoints between the Cartesian Points of a given vector (coordinates) and saves them in another (updatedCoordinates)  
//--------------------------------------------------------------------------------------------------------------------------------------------- 

std::vector<glm::dvec3> Plugin::generateMidPoint (std::vector <InfoStruct> const& structIn, float threshold, 
                                                    glm::vec3 earthRadius, std::shared_ptr<const cs::scene::CelestialObject> earth) { 

  std::vector <InfoStruct> totalStruct;

  std::vector<glm::dvec3> renderingVector;     

  for (int i=0; i < structIn.size()-1; i++) {     
    
    // the following definitions will just make the code syntax simpler 
    glm::dvec2 p1LongLatRadians = structIn[i].longLatRadians;
    glm::dvec2 p2LongLatRadians = structIn[i+1].longLatRadians;
    glm::dvec2 p1LongLatDegrees = structIn[i].longLatDegrees;
    glm::dvec2 p2LongLatDegrees = structIn[i+1].longLatDegrees;
    glm::dvec3 p1Cartesian      = structIn[i].Cartesian;
    glm::dvec3 p2Cartesian      = structIn[i+1].Cartesian;
    double p1Height = structIn[i].overSurfaceHeight;
    double p2Height = structIn[i+1].overSurfaceHeight;
    bool p1Bool = structIn[i].heightComesFromJson;
    bool p2Bool = structIn[i+1].heightComesFromJson;

    double distance = calculateDistance(structIn[i], structIn[i+1], earthRadius);

    if (distance > threshold) { 

      totalStruct.push_back(structIn[i]);

      int numMidPoints = static_cast<int> (distance/threshold);
      // glm::dvec3 segmentDirection = glm::normalize(secondPoint-firstPoint);
      // glm::dvec3 midPoint = firstPoint + (segmentDirection * (segmentLength * j)); 
      double segmentLength = distance/threshold; 

      std::vector <InfoStruct> midPointStruct;
      for (int j=1; j <= numMidPoints; j++) {
        InfoStruct temporaryStruct;
        // temporaryStruct.longLatRadians = glm::mix(p1LongLatRadians,p2LongLatRadians,segmentLength*j); 
        // temporaryStruct.longLatDegrees = glm::mix(p1LongLatDegrees,p2LongLatDegrees,segmentLength*j)
        
        temporaryStruct.Cartesian = glm::mix(p1Cartesian,p2Cartesian,(segmentLength*j)/distance); // fix length
        temporaryStruct.longLatRadians = cs::utils::convert::cartesianToLngLat(temporaryStruct.Cartesian, earthRadius);

        if (p1Bool || p2Bool) {
          temporaryStruct.overSurfaceHeight = ((p1Height + p2Height)/2);
          temporaryStruct.heightComesFromJson = true;
        } 
        else {
          temporaryStruct.overSurfaceHeight = earth->getSurface()->getHeight(temporaryStruct.longLatRadians);
          temporaryStruct.heightComesFromJson = false;
        }
        temporaryStruct.Cartesian = cs::utils::convert::toCartesian(temporaryStruct.longLatRadians, earthRadius, temporaryStruct.overSurfaceHeight);
        totalStruct.push_back(temporaryStruct);
        totalStruct.push_back(temporaryStruct);
      }
    }

    // in case the distance is so small that midPoints are not needed
    else {
      totalStruct.push_back(structIn[i]);
    } 
  }
  
  totalStruct.push_back(structIn[structIn.size() - 1]);

  for (int i=0; i<totalStruct.size(); i++) {
    
    // logger().info("{}: latLng({}, {}), h({}), cartLength({})", i, totalStruct[i].longLatDegrees.x, totalStruct[i].longLatDegrees.y, totalStruct[i].overSurfaceHeight, 
     //   glm::length(totalStruct[i].Cartesian) - std::min(std::min(earthRadius.x, earthRadius.y), earthRadius.z));
    renderingVector.push_back(totalStruct[i].Cartesian);
  }
return renderingVector;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// the function takes as an input the earthRadius and a couple of points (in Cartesian), whose Great Circle distance will be returned.
//------------------------------------------------------------------------------------------------------------------------------------ 
double Plugin::calculateDistance(InfoStruct const& p1, InfoStruct const& p2, glm::vec3 earthRadius) { 
  
  // great circle distance using Haversine formula     
  double rootArgument = cos(p1.longLatRadians[1]) * cos(p2.longLatRadians[1]) * cos(p1.longLatRadians[0] - p2.longLatRadians[0]) + sin(p1.longLatRadians[1]) * sin(p2.longLatRadians[1]);
  double averageEarthRadius = (earthRadius[0] + earthRadius[1] + earthRadius[2])/3;
  double distanceHaversine = (averageEarthRadius + (p1.overSurfaceHeight+p2.overSurfaceHeight)/2)  * acos(rootArgument);

  // staight-line distance
  double distanceStraight = glm::distance(p1.Cartesian,p2.Cartesian);
  
  //comparison
  if (distanceStraight>distanceHaversine) {
    return distanceStraight;
  }
  else {
    return distanceHaversine;    
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setWFSServer(std::string URL) {

  if (URL=="None") {
    mPointRenderer = nullptr; 
    mLineStringRenderer = nullptr;
    mPolygonRenderer = nullptr;
    return;
  }
  
  // TinyXML loading.
  //-----------------
  mBaseUrl = URL + "?SERVICE=WFS&VERSION=1.3.0";
  std::string mUrl = mBaseUrl + "&REQUEST=GetCapabilities";

  // Build the XML request
  std::stringstream xmlStream;      // String stream where the response will be stored
  curlpp::Easy      request;
  request.setOpt(curlpp::options::Url(mUrl));
  request.setOpt(curlpp::options::WriteStream(&xmlStream));
  request.setOpt(curlpp::options::NoSignal(true));
  request.setOpt(curlpp::options::SslVerifyPeer(false));

  // Execute the HTTP request and get the file
  try {
    request.perform();
  } catch (std::exception const& e) {
    std::stringstream message;
    message << "WFS capabilities request failed for '" << mUrl << "': '" << e.what() << "'";
    throw std::runtime_error(message.str());
  }

  // Transfer the info to a string (doc) and parse it
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

  // Save the selected content (in this case, names) of the XML GetCapabilities file
  std::stringstream ss;
  VistaXML::TiXmlElement* featureTypeElement = doc.RootElement() -> FirstChildElement("FeatureTypeList") -> FirstChildElement("FeatureType");
  while  (featureTypeElement) {
    const char* featureTypeName = featureTypeElement -> FirstChildElement ("Name") -> GetText();
    ss << featureTypeName << std::endl;
    featureTypes.push_back(featureTypeName);   

    featureTypeElement = featureTypeElement -> NextSiblingElement ("FeatureType");
  } /* logger().info(ss.str()); */

  // Clear the GUI list
  mGuiManager->getGui()->callJavascript(
  "CosmoScout.gui.clearDropdown", "wfsOverlays.setWFSFeatureType");
  
  // Set the GUI list
  mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wfsOverlays.setWFSFeatureType", "None", "None", false);

  for (int i=0; i < featureTypes.size(); i++) {
    mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wfsOverlays.setWFSFeatureType", featureTypes[i], featureTypes[i], false);
  }
  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setWFSFeatureType(std::string featureType) { 

  mPointRenderer = nullptr; 
  mLineStringRenderer = nullptr;
  mPolygonRenderer = nullptr;

  if (featureType == "None") {
    return;
  }

  std::string mFeatureUrl = mBaseUrl + "&outputFormat=json&REQUEST=GetFeature" + "&typeName=" + featureType;
  logger().info(mFeatureUrl);

  // JSON loading.
  //-----------------

  // Build the JSON request
  std::stringstream jsonStream;      // Where the response will be stored
  curlpp::Easy      jsonRequest;
  jsonRequest.setOpt(curlpp::options::Url(mFeatureUrl));
  jsonRequest.setOpt(curlpp::options::WriteStream(&jsonStream));
  jsonRequest.setOpt(curlpp::options::NoSignal(true));
  jsonRequest.setOpt(curlpp::options::SslVerifyPeer(false));

  // Execute the HTTP request and get the file
  try {
    jsonRequest.perform();            
  } catch (std::exception const& e) {
    std::stringstream message;
    message << "WFS capabilities request failed for '" << mFeatureUrl << "': '" << e.what() << "'";
    throw std::runtime_error(message.str());
  } /* logger().info(jsonStream.str()); */ 

  // Transfer the info to a string (doc) and parse it 
  std::string docString = jsonStream.str();
  nlohmann::json data = nlohmann::json::parse(docString);

  // Stores all the info for a single getCapabilities listed feature
  WFSFeatureCollection featureLocation;       

  from_json(data, featureLocation);

  // Saving all the attributes together in vectors before rendering
  //---------------------------------------------------------------
  // TODO: Likely to become a new function
  
  auto earth = mSolarSystem->getObject("Earth");
  glm::dvec3 earthRadius = earth->getRadii(); 
  
  std::vector <glm::dvec3> pointCoordinates;
  std::vector <glm::dvec3> lineStringCoordinates;
  std::vector <glm::dvec3> polygonCoordinates;

  logger().info("Size: {}", featureLocation.features.size());

  
  
 

  int numPoints = 0, numMultiPoints = 0, numLineStrings = 0, numMultiLineStrings = 0, numPolygons = 0, numMultiPolygons = 0; 
  
  for (int i = 0; i < featureLocation.features.size(); i++) {

    // checking "null" geometry (e.g. DWD -> dwd:Autowarn_Vorhersage)
    if (featureLocation.features[i].geometry == nullptr) {
      continue;
    }

    std::string type = featureLocation.features[i].geometry->mType;

    if (type == "Point") {
      numPoints++;
      std::shared_ptr<Point> point = std::dynamic_pointer_cast<Point>(featureLocation.features[i].geometry); 

      InfoStruct pointStruct;

      pointStruct.longLatDegrees = {point->mCoordinates[0], point->mCoordinates[1]};
      pointStruct.longLatRadians = {cs::utils::convert::toRadians(point->mCoordinates[0]),cs::utils::convert::toRadians(point->mCoordinates[1])};
      if (point->mCoordinates.size() > 2) {
        pointStruct.overSurfaceHeight = point->mCoordinates[2];

        if (pointStruct.overSurfaceHeight <= 0.0) {
          logger().info("{}", pointStruct.overSurfaceHeight);
        }

        pointStruct.heightComesFromJson = true;
      }
      else {                        
        pointStruct.overSurfaceHeight = earth->getSurface()->getHeight({pointStruct.longLatRadians[0], pointStruct.longLatRadians[1]}) + 10;
        pointStruct.heightComesFromJson = false;
      } 

      pointStruct.Cartesian = cs::utils::convert::toCartesian({pointStruct.longLatRadians[0], pointStruct.longLatRadians[1]}, earthRadius, pointStruct.overSurfaceHeight);

      pointCoordinates.push_back(pointStruct.Cartesian);
    }

    else if (type == "MultiPoint") {
      numMultiPoints++;
      std::shared_ptr<MultiPoint> multiPoint = std::dynamic_pointer_cast<MultiPoint>(featureLocation.features[i].geometry); 

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
      }   
    }

    else if (type == "LineString") {
      numLineStrings++;
      std::shared_ptr<LineString> lineString = std::dynamic_pointer_cast<LineString>(featureLocation.features[i].geometry); 
      std::vector<InfoStruct> lineStringStruct;

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

        lineStringStruct.push_back(temporaryStruct);
        if (i != 0 && i != lineString->mCoordinates.size()-1) {     
          lineStringStruct.push_back(temporaryStruct);
        } 
      }
      std::vector<glm::dvec3> lineStringVec = generateMidPoint(lineStringStruct, 100000.0, earthRadius, earth);
      // logger().info("{}, {}, {}", glm::length(line[0]), glm::length(line[line.size() / 2]), glm::length(line[line.size() - 1]));
      lineStringCoordinates.insert(lineStringCoordinates.end(), lineStringVec.begin(), lineStringVec.end());
    }
    
    else if (type == "MultiLineString") {
      numMultiLineStrings++; 
      std::shared_ptr<MultiLineString> multiLineString = std::dynamic_pointer_cast<MultiLineString>(featureLocation.features[i].geometry); 
      
      
      for (int i=0; i < multiLineString->mCoordinates.size(); i++) { 
std::vector<InfoStruct> multiLineStringStruct;
        for (int j=0; j < multiLineString->mCoordinates[i].size(); j++) { 
          InfoStruct temporaryStruct;
          // multiLineStringStruct.push_back({});
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
          
          multiLineStringStruct.push_back(temporaryStruct);
          if (j != 0 && j != multiLineString->mCoordinates[i].size()-1) {          
            multiLineStringStruct.push_back(temporaryStruct);
          } 
        } 
        std::vector<glm::dvec3> multiLineStringVec = generateMidPoint(multiLineStringStruct, 100000.0, earthRadius, earth);
      lineStringCoordinates.insert(lineStringCoordinates.end(), multiLineStringVec.begin(), multiLineStringVec.end());
      }
    } 
  
    else if (type == "Polygon") {
      numPolygons++; 
      std::shared_ptr<Polygon> polygon = std::dynamic_pointer_cast<Polygon>(featureLocation.features[i].geometry); 
      
      for (int i=0; i < polygon->mCoordinates.size(); i++) {
        
        // std::vector <glm::vec3> temporaryPolygonCoordinates;
        std::vector<InfoStruct> polygonStruct;
        for (int j=0; j < polygon->mCoordinates[i].size(); j++) {
          InfoStruct temporaryStruct;   
          // polygonStruct.push_back({});
          temporaryStruct.longLatDegrees = {polygon->mCoordinates[i][j][0], polygon->mCoordinates[i][j][1]};
          // polygonStruct[i].longLatRadians = {cs::utils::convert::toRadians(polygon->mCoordinates[i][j][0]), cs::utils::convert::toRadians(polygon->mCoordinates[i][j][1])};
          
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
          
          polygonStruct.push_back(temporaryStruct);
          if (j != 0 && j != polygon->mCoordinates[i].size()-1) {          
            polygonStruct.push_back(temporaryStruct);
          } 
        }
        std::vector<glm::dvec3> polygonVec = generateMidPoint(polygonStruct, 100000.0, earthRadius, earth);
      polygonCoordinates.insert(polygonCoordinates.end(), polygonVec.begin(), polygonVec.end());
      }      
    }

    else if (type == "MultiPolygon") {
      numMultiPolygons++; 
      std::shared_ptr<MultiPolygon> multiPolygon = std::dynamic_pointer_cast<MultiPolygon>(featureLocation.features[i].geometry);
      

      for (int i=0; i < multiPolygon->mCoordinates.size(); i++) { 

        for (int j=0; j < multiPolygon->mCoordinates[i].size(); j++) {
          // std::vector <glm::vec3> temporaryPolygonCoordinates;
          std::vector<InfoStruct> multiPolygonStruct;
          for (int k=0; k < multiPolygon->mCoordinates[i][j].size(); k++) {
            InfoStruct temporaryStruct;
            // multiPolygonStruct.push_back({});
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
            
            multiPolygonStruct.push_back(temporaryStruct);
            if (k != 0 && k != multiPolygon->mCoordinates[i][j].size()-1) {          
              multiPolygonStruct.push_back(temporaryStruct);
            } 
          }
        std::vector<glm::dvec3> multiPolygonVec = generateMidPoint(multiPolygonStruct, 100000.0, earthRadius, earth);
          polygonCoordinates.insert(polygonCoordinates.end(), multiPolygonVec.begin(), multiPolygonVec.end()); 
        }
      } 
        
    }
    
    else { 
      logger().warn(" {} data could not be rendered", type); 
    }
  }

  // Rendering
  //----------
  logger().info("Creating the renderers...");

  std::vector<glm::vec3> pointCoordinatesRendering;
  for (int i=0; i < pointCoordinates.size(); i++) {
    
      glm::vec3 coord = static_cast<glm::vec3>(pointCoordinates[i]);
      pointCoordinatesRendering.push_back(coord);
    
  }

  std::vector<glm::vec3> lineStringCoordinatesRendering;
  for (int i=0; i < lineStringCoordinates.size(); i++) {
    
      glm::vec3 coord = static_cast<glm::vec3>(lineStringCoordinates[i]);
      lineStringCoordinatesRendering.push_back(coord);
    
  }

  std::vector<glm::vec3> polygonCoordinatesRendering;
  for (int i=0; i < polygonCoordinates.size(); i++) {
    
      glm::vec3 coord = static_cast<glm::vec3>(polygonCoordinates[i]);
      polygonCoordinatesRendering.push_back(coord);
    
  }

  if (!pointCoordinates.empty()) {
    logger().info("points: {}, multiPoints: {}. Containing {} points.", numPoints, numMultiPoints, pointCoordinates.size());
    mPointRenderer = std::make_unique<FeatureRenderer>("Point", pointCoordinatesRendering, mSolarSystem, mAllSettings);
  }

  if (!lineStringCoordinates.empty()) {
    logger().info("lines: {}, multiLines: {}. Containing {} points.", numLineStrings, numMultiLineStrings, lineStringCoordinates.size()/2+1);
    mLineStringRenderer = std::make_unique<FeatureRenderer> ("LineString", lineStringCoordinatesRendering, mSolarSystem, mAllSettings);
  }

  if (!polygonCoordinates.empty()) {
    logger().info("polygons: {}, multiPolygons: {}. Containing {} points.", numPolygons, numMultiPolygons, polygonCoordinates.size()/2+1);
    mPolygonRenderer = std::make_unique<FeatureRenderer> ("Polygon", polygonCoordinatesRendering, mSolarSystem, mAllSettings);
  }

  if(!mPointRenderer && !mLineStringRenderer && !mPolygonRenderer) {
    logger().info("Server response: {}", jsonStream.str());
  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  // Setting a SideBar for the WFS Overlays Plugin
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "WFS Overlays", "device_hub", "../share/resources/gui/wfs_overlays_settings.html");
  mGuiManager->addPluginTabToSideBarFromHTML(
      "WFS Overlays", "device_hub", "../share/resources/gui/wfs_overlays_tab.html");
  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-wfs-overlays.js");

  // "Enable" callback via GUI
  mGuiManager->getGui()->registerCallback("wfsOverlays.setEnabled",
      "Enables or disables wfs overlays.",
      std::function([this](bool value) { mPluginSettings->mEnabled = value; }));    
  mPluginSettings->mEnabled.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("wfsOverlays.setEnabled", enable); });

  // "Set Server" callback via GUI
  mGuiManager->getGui()->registerCallback("wfsOverlays.setServer",
      "Set the current server to the one with the given name.",
      std::function([this](std::string&& name) {
        setWFSServer(name);
      }));
  mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wfsOverlays.setServer", "None", "None", false);

  // "Set Feature Type" callback via GUI
  mGuiManager->getGui()->registerCallback("wfsOverlays.setWFSFeatureType",
      "Set the current feature among all those provided by the server we chose.",
      std::function([this](std::string&& name) {
        logger().info("-----------------------------------------");
        logger().info("Selected new feature: {}", name);
        setWFSFeatureType(name);
      }));
  mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wfsOverlays.setWFSFeatureType", "None", "None", false);

  onLoad();

  // Show the list of servers
  std::vector<std::string> serverList = mPluginSettings->mWfs;
  for (int i=0; i < serverList.size(); i++) {
    mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wfsOverlays.setServer", serverList[i], serverList[i], false);
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



  