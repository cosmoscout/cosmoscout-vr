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
  cs::core::Settings::deserialize(j, "width", o.mWidth);
  cs::core::Settings::deserialize(j, "size", o.mSize);
}

void to_json(nlohmann::json& j, Plugin::Settings::Arrows const& o) {
  cs::core::Settings::serialize(j, "width", o.mWidth);
  cs::core::Settings::serialize(j, "size", o.mSize);
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
  
  for (auto const& settings : mPluginSettings->mArrows) {
    float angleX = 0.0f;
    glm::dvec3 rotAxisX(1.0f, 0.0f, 0.0f);
    glm::vec4 colorX(1.0f, 0.0f, 0.0f, 1.0f);
    float angleY = 90.0f;
    glm::dvec3 rotAxisY(0.0f, 0.0f, 1.0f);
    glm::vec4 colorY(0.0f, 1.0f, 0.0f, 1.0f);
    float angleZ = -90.0f;
    glm::dvec3 rotAxisZ(0.0f, 1.0f, 0.0f);
    glm::vec4 colorZ(0.0f, 0.0f, 1.0f, 1.0f);

    std::vector<float> lineVertices = createArrowVertices();

    logger().info("Arrow width for this one is {}", settings.second.mWidth);

    auto arrowX = std::make_shared<Arrow>(mPluginSettings, mSolarSystem, lineVertices, rotAxisX, angleX, colorX, settings.second.mWidth, settings.second.mSize);
    arrowX->setParentName(settings.first);
    auto arrowY = std::make_shared<Arrow>(mPluginSettings, mSolarSystem, lineVertices, rotAxisY, angleY, colorY, settings.second.mWidth, settings.second.mSize);
    arrowY->setParentName(settings.first);
    auto arrowZ = std::make_shared<Arrow>(mPluginSettings, mSolarSystem, lineVertices, rotAxisZ, angleZ, colorZ, settings.second.mWidth, settings.second.mSize);
    arrowZ->setParentName(settings.first);

    std::vector<std::shared_ptr<Arrow>> arrowGroup = {arrowX, arrowY, arrowZ};

    mArrows.emplace(settings.first, arrowGroup);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> Plugin::createArrowVertices() {
  std::vector<float> vertices;

  /*const int SEGMENTS = 16;
  const float PI = 3.141592653f;

  float shaftRadius = 10.02f;
  float shaftLength = 10.08f;
  float coneRadius = 10.06f;
  float coneLength = 10.02f;

  // Shaft cylinder as arrowtrail
  for (int i = 0; i < SEGMENTS; i++) {
    float angle1 = (float)i / SEGMENTS * 2.0f * PI;
    float angle2 = (float)(i + 1) / SEGMENTS * 2.0f * PI;

    float x1 = cos(angle1) * shaftRadius;
    float x2 = cos(angle2) * shaftRadius;
    float z1 = sin(angle1) * shaftRadius;
    float z2 = sin(angle2) * shaftRadius;

    // two triangles
    vertices.insert(vertices.end(), {x1, 0.0f, z1, x2, 0.0f, z2, x1, shaftLength, z1});
    vertices.insert(vertices.end(), {x2, 0.0f, z2, x2, shaftLength, z2, x1, shaftLength, z1});
  }

  // Cone as arrowhead
  for (int i = 0; i < SEGMENTS; i++) {
    float angle1 = (float)i / SEGMENTS * 2.0f * PI;
    float angle2 = (float)(i + 1) / SEGMENTS * 2.0f * PI;

    float x1 = cos(angle1) * coneRadius;
    float x2 = cos(angle2) * coneRadius;

    float z1 = sin(angle1) * coneRadius;
    float z2 = sin(angle2) * coneRadius;
  }*/

  vertices = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  return vertices;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::coordinatearrows
