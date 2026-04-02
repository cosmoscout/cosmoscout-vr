////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "Arrow.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/ObjLoader.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/VistaSystem.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::coordinatearrows::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::coordinatearrows {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "arrows", o.mArrows);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "arrows", o.mArrows);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Arrows& o) {
  cs::core::Settings::deserialize(j, "size", o.mSize);
  cs::core::Settings::deserialize(j, "showX", o.mShowX);
  cs::core::Settings::deserialize(j, "showY", o.mShowY);
  cs::core::Settings::deserialize(j, "showZ", o.mShowZ);
}

void to_json(nlohmann::json& j, Plugin::Settings::Arrows const& o) {
  cs::core::Settings::serialize(j, "size", o.mSize);
  cs::core::Settings::serialize(j, "showX", o.mShowX);
  cs::core::Settings::serialize(j, "showY", o.mShowY);
  cs::core::Settings::serialize(j, "showZ", o.mShowZ);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-coordinate-arrows"] = *mPluginSettings; });

  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-coordinate-arrows.js");
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Coordinate Arrows", "label_off", "../share/resources/gui/coordinate-arrows-settings.html");

  mGuiManager->getGui()->registerCallback("coordinateArrows.enableArrows",
    "Enables or disables the rendering of the arrows.",
    std::function([this](bool value) {
      mPluginSettings->mEnableArrows = value;
    }));
  mPluginSettings->mEnableArrows.connectAndTouch([this](bool value) {
    mGuiManager->setCheckboxValue("coordinateArrows.enableArrows", true);
  });

  // Load settings
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mGuiManager->removePluginTab("Coordinate Arrows");
  mGuiManager->removeSettingsSection("Coordinate Arrows");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-coordinate-arrows"), *mPluginSettings);
  std::shared_ptr<cs::graphics::ObjLoader> arrowModel = std::make_shared<cs::graphics::ObjLoader>("../share/resources/models/arrow.obj");
  
  // Adds a group of arrows for every object stated in settings.
  for (auto const& settings : mPluginSettings->mArrows) {

    // Define the angles and axis the base arrow model is rotated.
    // Base model faces positive x-direction and is thus not rotated for visualizing the X-axis.

    // Dont rotate the base model and make it red.
    float angleX = 0.0f;
    glm::dvec3 rotAxisX(1.0f, 0.0f, 0.0f);
    glm::vec4 colorX(1.0f, 0.0f, 0.0f, 1.0f);

    // Rotate the base model to make it face the Y-axis and make it green.
    float angleY = 90.0f;
    glm::dvec3 rotAxisY(0.0f, 0.0f, 1.0f);
    glm::vec4 colorY(0.0f, 1.0f, 0.0f, 1.0f);

    // Rotate the base model to make it face the Z-axis and make it blue.
    float angleZ = -90.0f;
    glm::dvec3 rotAxisZ(0.0f, 1.0f, 0.0f);
    glm::vec4 colorZ(0.0f, 0.0f, 1.0f, 1.0f);

    // Sumarizes the arrows of the three axises in a vector as "a group of arrows",
    // representing the whole coordinate visualiztion for an object.
    std::vector<std::shared_ptr<Arrow>> arrowGroup;
    logger().info("-------------------PFEILE NOCH NICHT ERSTELLT!!!!!");
    // Creates the arrows for the X, Y and Z-axis and sets them to the visualize the object stated in settings.
    if (settings.second.mShowX) {
      logger().info("-------------------BIN JETZT HIER!!");
      auto arrowX = std::make_shared<Arrow>(mPluginSettings, mSolarSystem, arrowModel, rotAxisX, angleX, colorX, settings.second.mSize * 0.1f);
      arrowX->setParentName(settings.first);
      arrowGroup.push_back(arrowX);
    }
    if (settings.second.mShowY) {
      auto arrowY = std::make_shared<Arrow>(mPluginSettings, mSolarSystem, arrowModel, rotAxisY, angleY, colorY, settings.second.mSize * 0.1f);
      arrowY->setParentName(settings.first);
      arrowGroup.push_back(arrowY);
    }
    if (settings.second.mShowZ) {
      auto arrowZ = std::make_shared<Arrow>(mPluginSettings, mSolarSystem, arrowModel, rotAxisZ, angleZ, colorZ, settings.second.mSize * 0.1f);
      arrowZ->setParentName(settings.first);
      arrowGroup.push_back(arrowZ);
    }
 
    mArrows.emplace(settings.first, arrowGroup);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::coordinatearrows
