////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "Arrow.hpp"
#include "Axis.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"


////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::orientationtools::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::orientationtools {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "arrows", o.mArrows);
  cs::core::Settings::deserialize(j, "axes", o.mAxes);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "arrows", o.mArrows);
  cs::core::Settings::serialize(j, "axes", o.mAxes);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Arrows& o) {
  cs::core::Settings::deserialize(j, "size", o.mSize);
  cs::core::Settings::deserialize(j, "disableX", o.mDisableX);
  cs::core::Settings::deserialize(j, "disableY", o.mDisableY);
  cs::core::Settings::deserialize(j, "disableZ", o.mDisableZ);
}

void to_json(nlohmann::json& j, Plugin::Settings::Arrows const& o) {
  cs::core::Settings::serialize(j, "size", o.mSize);
  cs::core::Settings::serialize(j, "disableX", o.mDisableX);
  cs::core::Settings::serialize(j, "disableY", o.mDisableY);
  cs::core::Settings::serialize(j, "disableZ", o.mDisableZ);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Axis& o) {
  cs::core::Settings::deserialize(j, "size", o.mSize);
  cs::core::Settings::deserialize(j, "color", o.mColor);
  cs::core::Settings::deserialize(j, "disableX", o.mDisableX);
  cs::core::Settings::deserialize(j, "disableY", o.mDisableY);
  cs::core::Settings::deserialize(j, "disableZ", o.mDisableZ);
}

void to_json(nlohmann::json& j, Plugin::Settings::Axis const& o) {
  cs::core::Settings::serialize(j, "size", o.mSize);
  cs::core::Settings::serialize(j, "color", o.mColor);
  cs::core::Settings::serialize(j, "disableX", o.mDisableX);
  cs::core::Settings::serialize(j, "disableY", o.mDisableY);
  cs::core::Settings::serialize(j, "disableZ", o.mDisableZ);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-orientation-tools"] = *mPluginSettings; });

  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-orientation-tools.js");
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Orientation Tools", "label_off", "../share/resources/gui/orientation-tools-settings.html");

  mGuiManager->getGui()->registerCallback("orientationTools.enableArrows",
    "Enables or disables the rendering of the arrows.",
    std::function([this](bool value) {
      mPluginSettings->mEnableArrows = value;
    }));

  mGuiManager->getGui()->registerCallback("orientationTools.enableAxes",
    "Enables or disables the rendering of the axes.",
    std::function([this](bool value) {
      mPluginSettings->mEnableAxes = value;
    }));

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mGuiManager->removePluginTab("Orientation Tools");
  mGuiManager->removeSettingsSection("Orientation Tools");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-orientation-tools"), *mPluginSettings);
  
  // Adds a group of arrows for every object stated in settings.
  for (auto const& settings : mPluginSettings->mArrows) {
    addArrowsGroup(settings);
  }

  // Adds an axis for every object stated in settings.
  for (auto const& settings : mPluginSettings->mAxes) {
    addAxesGroup(settings);
  }
  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::addArrowsGroup(std::pair<const std::string, csp::orientationtools::Plugin::Settings::Arrows> settings) {
    // Sumarizes the arrows of the three axises in a vector as "a group of arrows",
    // representing the whole coordinate visualiztion for an object.
    std::vector<std::shared_ptr<Arrow>> arrowGroup;

    // Creates the arrows for the X, Y and Z-axis and sets them to the visualize the object stated in settings.
    // Only does so if no setting is set to disable that.
    if (!(settings.second.mDisableX.has_value() && settings.second.mDisableX.value())) {
      auto arrowX = std::make_shared<Arrow>(mPluginSettings, mSolarSystem, mArrowModel, mRotAxisX, mAngleX, mColorX, settings.second.mSize * 0.1f);
      arrowX->setParentName(settings.first);
      arrowGroup.push_back(arrowX);
    }
    if (!(settings.second.mDisableY.has_value() && settings.second.mDisableY.value())) {
      auto arrowY = std::make_shared<Arrow>(mPluginSettings, mSolarSystem, mArrowModel, mRotAxisY, mAngleY, mColorY, settings.second.mSize * 0.1f);
      arrowY->setParentName(settings.first);
      arrowGroup.push_back(arrowY);
    }
    if (!(settings.second.mDisableZ.has_value() && settings.second.mDisableZ.value())) {
      auto arrowZ = std::make_shared<Arrow>(mPluginSettings, mSolarSystem, mArrowModel, mRotAxisZ, mAngleZ, mColorZ, settings.second.mSize * 0.1f);
      arrowZ->setParentName(settings.first);
      arrowGroup.push_back(arrowZ);
    }
 
    mArrows.emplace(settings.first, arrowGroup);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::addAxesGroup(std::pair<const std::string, csp::orientationtools::Plugin::Settings::Axis> settings) {
  // Summarizes the three axes in a vector as "a group of axes",
  // representing the whole axis visualization for an object.
  std::vector<std::shared_ptr<Axis>> axisGroup;

  // Creates two axes for each axis visualization. Rotates one axis to fit on the axis it shall represent and the other 180° in the other direction.
  if (!(settings.second.mDisableX.has_value() && settings.second.mDisableX.value())) {
    auto axis = std::make_shared<Axis>(mPluginSettings, mSolarSystem, mAxisModel, mRotAxisX, mAngleX, glm::vec4(settings.second.mColor, 1.F), settings.second.mSize * 0.1f);
    axis->setParentName(settings.first);
    axisGroup.push_back(axis);
  }
  
  if (!(settings.second.mDisableY.has_value() && settings.second.mDisableY.value())) {
    auto axis = std::make_shared<Axis>(mPluginSettings, mSolarSystem, mAxisModel, mRotAxisY, mAngleY, glm::vec4(settings.second.mColor, 1.F), settings.second.mSize * 0.1f);
    axis->setParentName(settings.first);
    axisGroup.push_back(axis);
  }

  if (!(settings.second.mDisableZ.has_value() && settings.second.mDisableZ.value())) {
    auto axis = std::make_shared<Axis>(mPluginSettings, mSolarSystem, mAxisModel, mRotAxisZ, mAngleZ, glm::vec4(settings.second.mColor, 1.F), settings.second.mSize * 0.1f);
    axis->setParentName(settings.first);
    axisGroup.push_back(axis);
  }

  mAxes.emplace(settings.first, axisGroup);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::orientationtools
