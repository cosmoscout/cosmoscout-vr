////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "SimpleObject.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-utils/filesystem.hpp"

#include <regex>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::simpleobjects::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::simpleobjects {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::SimpleObject& o) {
  cs::core::Settings::deserialize(j, "modelFile", o.mModelFile);
  cs::core::Settings::deserialize(j, "environmentMap", o.mEnvironmentMap);
  cs::core::Settings::deserialize(j, "anchor", o.mAnchorName);
  cs::core::Settings::deserialize(j, "lngLat", o.mLngLat);
  cs::core::Settings::deserialize(j, "rotation", o.mRotation);
  cs::core::Settings::deserialize(j, "alignToSurface", o.mAlignToSurface);
  cs::core::Settings::deserialize(j, "elevation", o.mElevation);
  cs::core::Settings::deserialize(j, "scale", o.mScale);
  cs::core::Settings::deserialize(j, "diagonalLength", o.mDiagonalLength);
}

void to_json(nlohmann::json& j, Plugin::Settings::SimpleObject const& o) {
  cs::core::Settings::serialize(j, "modelFile", o.mModelFile);
  cs::core::Settings::serialize(j, "environmentMap", o.mEnvironmentMap);
  cs::core::Settings::serialize(j, "anchor", o.mAnchorName);
  cs::core::Settings::serialize(j, "lngLat", o.mLngLat);
  cs::core::Settings::serialize(j, "rotation", o.mRotation);
  cs::core::Settings::serialize(j, "alignToSurface", o.mAlignToSurface);
  cs::core::Settings::serialize(j, "elevation", o.mElevation);
  cs::core::Settings::serialize(j, "scale", o.mScale);
  cs::core::Settings::serialize(j, "diagonalLength", o.mDiagonalLength);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "objects", o.mSimpleObjects);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "objects", o.mSimpleObjects);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

// Initializing GUI elements

  mGuiManager->addCssToGui("css/csp-simple-objects.css");
  mGuiManager->addHtmlToGui("simple-objects-editor", "../share/resources/gui/simple-objects-editor.html");
  mGuiManager->addHtmlToGui("simple-objects-list-item-template", "../share/resources/gui/simple-objects-templates.html");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-simple-objects-editor.js");
  mGuiManager->addPluginTabToSideBarFromHTML("Objects", "view_in_ar", "../share/resources/gui/simple-objects-tab.html");


// -------- Initializing GUI callback functions ------------------------

  /**
   * This function is called when a edit button in the object list is clicked.
   * It sends the corresponding json config to the gui which loads it into the editor 
   * and creates a new temporary object which displays the changes entered in the editor.
   * The object that is being edited is hidden until the changes are saved / discarded.
   */
  
  mGuiManager->getGui()->registerCallback("simpleObjects.edit",
    "Edits the settings of the simple object with given ID.",
    std::function([this](std::string&& objectName) {

    logger().debug("Edit called for object '{}'", objectName);

    auto object = mPluginSettings.mSimpleObjects.find(objectName);
    if (object != mPluginSettings.mSimpleObjects.end()) {
      nlohmann::json json = object->second;
      mGuiManager->getGui()->callJavascript("CosmoScout.simpleObjectsEditor.edit", objectName, json.dump());

      auto it = std::find_if(mSimpleObjects.begin(), mSimpleObjects.end(), 
              [&objectName](const std::shared_ptr<csp::simpleobjects::SimpleObject>& obj) {
                return obj->getName() == objectName;
              });

      if (it != mSimpleObjects.end()) {
        tmpSimpleObject = std::make_shared<SimpleObject>(objectName, object->second, mSceneGraph, mAllSettings, mSolarSystem);
        tmpSimpleObject->setSun(mSolarSystem->getSun());
        tmpSimpleObject->update();

        (*it)->setEditEnabled(true);
      }
      
    } else {
      logger().warn("Failed to execute 'simpleObjects.edit' for object '{}': No such object registered!", objectName);
    }
  }));


  /**
   * The update callback is called as soon as some changes in the editor are made. 
   * It will display them by writing the new configuration to the temporary object. 
   * If the temporary object does not exist, it will be created with the new configuration.
   */

  mGuiManager->getGui()->registerCallback("simpleObjects.update",
    "Generates a new temporary simple object to display if it does not exist yet or updates the settings otherwise.",
    std::function([this](std::string&& name, std::string&& jsonString) {
    
    Settings::SimpleObject settings;
    auto json = nlohmann::json::parse(jsonString);
    json.get_to(settings);

    if(tmpSimpleObject == nullptr) {
      auto simpleobject = std::make_shared<SimpleObject>(name, settings, mSceneGraph, mAllSettings, mSolarSystem);
      simpleobject->setSun(mSolarSystem->getSun());
      tmpSimpleObject = std::move(simpleobject);
    } else {
      tmpSimpleObject->updateConfig(name, settings);
    }
  }));


  /**
   * Saves the configuration to the object with the given name 'newName' and deletes the temporary object.
   * If the name didn't change, the configuration of the object will be updated accordingly.
   * Otherwise the old one will be deleted and an object with the new name and new config will be created.
   */

  mGuiManager->getGui()->registerCallback("simpleObjects.save",
    "Adds a new simple object to the scene. If a temporary object with the same name exists, it will be deleted.",
    std::function([this](std::string&& oldName, std::string&& newName, std::string&& jsonString) {
    
    logger().debug("Save called. Old Name: '{}' New Name: '{}'", oldName, newName);

    Settings::SimpleObject settings;
    auto json = nlohmann::json::parse(jsonString);
    json.get_to(settings);

    if(oldName == newName) {
      mPluginSettings.mSimpleObjects[oldName] = settings;
      
      auto it = std::find_if(mSimpleObjects.begin(), mSimpleObjects.end(), 
            [&oldName](const std::shared_ptr<csp::simpleobjects::SimpleObject>& obj) {
              return obj->getName() == oldName;
            });

      (*it)->updateConfig(oldName, settings);
      (*it)->setEditEnabled(false);

    } else {
      
      addObject(newName, settings);
      mPluginSettings.mSimpleObjects[newName] = settings;

      if(oldName != "") {
        mSimpleObjects.erase(
          std::remove_if(
            mSimpleObjects.begin(), 
            mSimpleObjects.end(),
            [&](std::shared_ptr<SimpleObject> const & so) { return so->getName() == oldName; }
          ), 
          mSimpleObjects.end()
        );      

        mPluginSettings.mSimpleObjects.erase(oldName);
        mGuiManager->getGui()->callJavascript("CosmoScout.simpleObjectsEditor.removeObjectFromList", oldName);
      }

    }

    tmpSimpleObject.reset();
  }));
  

  /**
   * This function is called when the close button of the editor is clicked.
   * In this case no changes should be made, so the temporary object just needs to be removed and 
   * the normal object without any changes will be displayed again.
   */

  mGuiManager->getGui()->registerCallback("simpleObjects.undoEdit",
    "Reverts all changes of the edited object with given name. (Disables the edit mode and deletes the temporary object.)",
    std::function([this](std::string&& objectName) {
    
    logger().debug("UndoEdit called for object '{}'", objectName);

    auto it = std::find_if(mSimpleObjects.begin(), mSimpleObjects.end(), 
            [&objectName](const std::shared_ptr<csp::simpleobjects::SimpleObject>& obj) {
              return obj->getName() == objectName;
            });

    if (it != mSimpleObjects.end()) {
      (*it)->setEditEnabled(false);
    }

    tmpSimpleObject.reset();
  }));


  /**
   * The remove callback function deletes the object with given name und clears the temporary object.
   */

  mGuiManager->getGui()->registerCallback("simpleObjects.remove",
    "Removes the simple object with the given ID.",
    std::function([this](std::string&& objectName) {
    
    logger().debug("Remove called for object '{}'", objectName);

    mPluginSettings.mSimpleObjects.erase(objectName);
    mSimpleObjects.erase(
      std::remove_if(
          mSimpleObjects.begin(), 
          mSimpleObjects.end(),
          [&](std::shared_ptr<SimpleObject> const & so) { return so->getName() == objectName; }
      ), 
      mSimpleObjects.end()
    );
    tmpSimpleObject.reset();
  }));


  mGuiManager->getGui()->registerCallback("simpleObjects.setPickLocationEnabled",
    "Toggles the ability to pick a location on the ground that will be set in the simple objects editor",
    std::function([this](bool enable) { pickLocationToolEnabled = enable; }));

  

  // TODO: the next 3 callback functions are not used. 
  // Removing them does not work because the gui needs them as default callback functions?!
  // Did not search for a way to remove callback entry from the button divs yet.

  mGuiManager->getGui()->registerCallback("simpleObjects.setAlignToSufaceEnabled", 
    "Sets whether the current object should be aligned to the surface.", 
    std::function([this](bool enable) { alignToSurfaceEnabled = enable; }));

  mGuiManager->getGui()->registerCallback("simpleObjects.setModelFile", 
    "Sets the file name of the gltf model file.",
    std::function([this](std::string&& model) { modelFile = model; }));
  
  mGuiManager->getGui()->registerCallback("simpleObjects.setEnvironmentMap",
    "Sets the file name of the environment map.", 
    std::function([this](std::string&& map) { environmentMap = map; }));




  /**
   * When the pick location functionality is enabled, this function checks for an intersection with a CelestialBody 
   * in the free FOV where no GUI elements are.
   * If available, the coordinates at the clicked location and the SPICE center are send to the GUI.
   */ 

  mOnClickConnection = mInputManager->pButtons[0].connect([this](bool pressed) {
    if (!pressed && !mInputManager->pHoveredGuiItem.get() && pickLocationToolEnabled) {
      auto intersection = mInputManager->pHoveredObject.get().mObject;

      if (!intersection) {
        return;
      }

      auto body = std::dynamic_pointer_cast<cs::scene::CelestialBody>(intersection);

      if (!body) {
        return;
      }

      auto radii = body->getRadii();
      auto lngLat = cs::utils::convert::toDegrees(cs::utils::convert::cartesianToLngLat(mInputManager->pHoveredObject.get().mPosition, radii));
      mGuiManager->getGui()->callJavascript("CosmoScout.simpleObjectsEditor.setLngLatAnchor", lngLat[0], lngLat[1], body->getCenterName());       
    }
  });

  
// Find model and environment map files and add them to the dropdown menus.

  logger().info("Scanning for model files..");
  initDropdown("simpleObjects.setModelFile", "../share/resources/models", std::regex(".+(\\.gltf|\\.glb)$"));

  logger().info("Scanning for environment maps..");
  initDropdown("simpleObjects.setEnvironmentMap", "../share/resources/textures", std::regex(".+(\\.dds)$"));


// Load and place all objects from the settings.json


  mPluginSettings = mAllSettings->mPlugins.at("csp-simple-objects");

  if(mPluginSettings.mSimpleObjects.size() > 0) {
    logger().info("Loading objects from settings..");
  } else { 
    logger().warn("No objects configured in the settings.");
  }
  
  for (auto const& objSettings : mPluginSettings.mSimpleObjects) {
    
    logger().debug("  * {}", objSettings.first);

    // check whether the specified anchor exists
    auto anchor = mAllSettings->mAnchors.find(objSettings.second.mAnchorName);
    if (anchor == mAllSettings->mAnchors.end()) {
      throw std::runtime_error(
          "There is no Anchor \"" + objSettings.first + "\" defined in the settings.");
    }
    addObject(objSettings.first, objSettings.second);
  }

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mGuiManager->removePluginTab("Objects");

  mGuiManager->getGui()->unregisterCallback("simpleObjects.setPickLocationEnabled");
  mGuiManager->getGui()->unregisterCallback("simpleObjects.setAlignToSufaceEnabled");
  mGuiManager->getGui()->unregisterCallback("simpleObjects.setModelFile");
  mGuiManager->getGui()->unregisterCallback("simpleObjects.setEnvironmentMap");
  mGuiManager->getGui()->unregisterCallback("simpleObjects.edit");
  mGuiManager->getGui()->unregisterCallback("simpleObjects.update");
  mGuiManager->getGui()->unregisterCallback("simpleObjects.save");
  mGuiManager->getGui()->unregisterCallback("simpleObjects.undoEdit");
  mGuiManager->getGui()->unregisterCallback("simpleObjects.remove");

  mGuiManager->getGui()->callJavascript("CosmoScout.gui.unregisterHtml", "simple-objects-editor");
  mGuiManager->getGui()->callJavascript("CosmoScout.gui.unregisterHtml", "simple-objects-list-item-template");
  mGuiManager->getGui()->callJavascript("CosmoScout.gui.unregisterCss", "css/csp-simple-objects.css");
  
  mInputManager->pButtons[0].disconnect(mOnClickConnection);
  
  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  for(std::shared_ptr<SimpleObject> so : mSimpleObjects) {
    so->update();
  }

  if(tmpSimpleObject != nullptr) {
    tmpSimpleObject->update();
  }

  mGuiManager->getGui()->callJavascript("CosmoScout.simpleObjectsEditor.updatePickLocationButton");
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::initDropdown(const std::string jsFunction, const std::string folder, const std::regex pattern) {
  auto files(cs::utils::filesystem::listFiles(folder, pattern));

  for (auto const& file : files) {
    std::string name(file);
    logger().debug("  * {}", name);

    const size_t lastSlashIdx = name.find_last_of("\\/");
    if (std::string::npos != lastSlashIdx) {
      name.erase(0, lastSlashIdx + 1);
    }

    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.addDropdownValue", jsFunction, name, name, false);
  }
}

void Plugin::addObject(std::string name, Settings::SimpleObject settings) {

  auto simpleobject = std::make_shared<SimpleObject>(name, settings, mSceneGraph, mAllSettings, mSolarSystem);

  simpleobject->setSun(mSolarSystem->getSun());
    
  mGuiManager->getGui()->callJavascript("CosmoScout.simpleObjectsEditor.addObjectToList", name);

  mSimpleObjects.push_back(simpleobject);
}

} // namespace csp::satellites