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
#include "logger.hpp"

#include <iostream>

#include <VistaTools/tinyXML/tinyxml.h>

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <regex>

#include <boost/filesystem.hpp>

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

// Here, we are binding the components of the Settings struct (from the Plugin class)
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
//----------------------------------------------------------------------------

void from_json(const nlohmann::json& j, std::shared_ptr<GeometryBase>& o) {
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

void Plugin::setWFSServer(std::string URL) {

  if (URL=="None") {
    mRenderers.clear();
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

  if (featureType == "None") {
    mRenderers.clear();
    return;
  }

  std::string mFeatureUrl = mBaseUrl + "&outputFormat=json&REQUEST=GetFeature" + "&typeName=" + featureType;
  // logger().info(mFeatureUrl);

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

  // Rendering
  //----------
  // TODO: Likely to become a new function
  
  mRenderers.clear();

  for (int i; i < featureLocation.features.size(); i++) {
    std::string type = featureLocation.features[i].geometry->mType;

    if (type == "Point") {
      std::shared_ptr<Point> point = std::dynamic_pointer_cast<Point>(featureLocation.features[i].geometry); 
      mRenderers.push_back(std::make_unique<FeatureRenderer>(point, mSolarSystem));
    }

    if (type == "MultiPoint") {
      std::shared_ptr<MultiPoint> multiPoint = std::dynamic_pointer_cast<MultiPoint>(featureLocation.features[i].geometry); 
      mRenderers.push_back(std::make_unique<FeatureRenderer>(multiPoint, mSolarSystem));
    }

    if (type == "LineString") {
      std::shared_ptr<LineString> lineString = std::dynamic_pointer_cast<LineString>(featureLocation.features[i].geometry); 
      mRenderers.push_back(std::make_unique<FeatureRenderer>(lineString, mSolarSystem));
    }

    if (type == "MultiLineString") {
      std::shared_ptr<MultiLineString> multiLineString = std::dynamic_pointer_cast<MultiLineString>(featureLocation.features[i].geometry); 
      mRenderers.push_back(std::make_unique<FeatureRenderer>(multiLineString, mSolarSystem));
    }

    if (type == "Polygon") {
      std::shared_ptr<Polygon> polygon = std::dynamic_pointer_cast<Polygon>(featureLocation.features[i].geometry); 
      mRenderers.push_back(std::make_unique<FeatureRenderer>(polygon, mSolarSystem));
    }

    if (type == "MultiPolygon") {
      std::shared_ptr<MultiPolygon> multiPolygon = std::dynamic_pointer_cast<MultiPolygon>(featureLocation.features[i].geometry); 
      mRenderers.push_back(std::make_unique<FeatureRenderer>(multiPolygon, mSolarSystem));
    }

logger().info(" renderers created: {} ", mRenderers.size());

/* for (int i=0; i < featureLocation.features.size(); i++) {
  logger().info("Element: {}: ID: {} Coordinates: ({},{})", i, featureLocation.features[i].id, featureLocation.features[i].geometry.coordinates[0], featureLocation.features[i].geometry.coordinates[1]);
} */ 

/* std::array<float, 2> iCoordinates = mRenderer->getCoordinates(i);
  logger().info("Element: {} Coordinates: ({}, {})", i, iCoordinates[0], iCoordinates[1]);
} */
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



  