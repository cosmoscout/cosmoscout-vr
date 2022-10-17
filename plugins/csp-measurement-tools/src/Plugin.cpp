////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/logger.hpp"

#include "../../csl-tools/src/Mark.hpp"

#include "logger.hpp"

#include "DipStrikeTool.hpp"
#include "EllipseTool.hpp"
#include "PathTool.hpp"
#include "PolygonTool.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::measurementtools::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::measurementtools {

namespace {
// These are only used during settings loading, as they are required in the free from_json methods.
// Loading never happens on multiple threads, so this is a save thing to do.
std::shared_ptr<cs::core::InputManager> sInputManager;
std::shared_ptr<cs::core::SolarSystem>  sSolarSystem;
std::shared_ptr<cs::core::Settings>     sSettings;
std::shared_ptr<cs::core::TimeControl>  sTimeControl;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void updateTools(T& tools) {
  for (auto it = tools.begin(); it != tools.end();) {
    if ((*it)->pShouldDelete.get()) {
      it = tools.erase(it);
    } else {
      (*it)->update();
      ++it;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void deserializeTools(nlohmann::json const& j, std::string const& name, T& tools) {
  auto array = j.find(name);
  if (array != j.end()) {
    tools.resize(array->size());
    for (size_t i(0); i < tools.size(); ++i) {
      array->at(i).get_to(tools[i]);
    }
  } else {
    tools.clear();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, std::shared_ptr<DipStrikeTool>& o) {
  if (!o) {
    o = std::make_shared<DipStrikeTool>(sInputManager, sSolarSystem, sSettings, "");
  }

  std::string object;
  cs::core::Settings::deserialize(j, "object", object);
  o->setObjectName(object);

  cs::core::Settings::deserialize(j, "color", o->pColor);
  cs::core::Settings::deserialize(j, "scaleDistance", o->pScaleDistance);
  cs::core::Settings::deserialize(j, "size", o->pSize);
  cs::core::Settings::deserialize(j, "opacity", o->pOpacity);
  cs::core::Settings::deserialize(j, "text", o->pText);

  std::vector<glm::dvec2> positions;
  cs::core::Settings::deserialize(j, "positions", positions);
  o->setPositions(positions);
}

void to_json(nlohmann::json& j, std::shared_ptr<DipStrikeTool> const& o) {
  cs::core::Settings::serialize(j, "object", o->getObjectName());
  cs::core::Settings::serialize(j, "color", o->pColor);
  cs::core::Settings::serialize(j, "scaleDistance", o->pScaleDistance);
  cs::core::Settings::serialize(j, "size", o->pSize);
  cs::core::Settings::serialize(j, "opacity", o->pOpacity);
  cs::core::Settings::serialize(j, "text", o->pText);
  cs::core::Settings::serialize(j, "positions", o->getPositions());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, std::shared_ptr<EllipseTool>& o) {
  if (!o) {
    o = std::make_shared<EllipseTool>(sInputManager, sSolarSystem, sSettings, "");
  }

  std::string object;
  cs::core::Settings::deserialize(j, "object", object);
  o->setObjectName(object);

  cs::core::Settings::deserialize(j, "handle0", o->getCenterHandle().pLngLat);
  cs::core::Settings::deserialize(j, "handle1", o->getFirstHandle().pLngLat);
  cs::core::Settings::deserialize(j, "handle2", o->getSecondHandle().pLngLat);
  cs::core::Settings::deserialize(j, "color", o->pColor);
  cs::core::Settings::deserialize(j, "scaleDistance", o->getCenterHandle().pScaleDistance);
  cs::core::Settings::deserialize(j, "text", o->getCenterHandle().pText);
  cs::core::Settings::deserialize(j, "minimized", o->getCenterHandle().pMinimized);
}

void to_json(nlohmann::json& j, std::shared_ptr<EllipseTool> const& o) {
  cs::core::Settings::serialize(j, "object", o->getObjectName());
  cs::core::Settings::serialize(j, "handle0", o->getCenterHandle().pLngLat);
  cs::core::Settings::serialize(j, "handle1", o->getFirstHandle().pLngLat);
  cs::core::Settings::serialize(j, "handle2", o->getSecondHandle().pLngLat);
  cs::core::Settings::serialize(j, "color", o->pColor);
  cs::core::Settings::serialize(j, "scaleDistance", o->getCenterHandle().pScaleDistance);
  cs::core::Settings::serialize(j, "text", o->getCenterHandle().pText);
  cs::core::Settings::serialize(j, "minimized", o->getCenterHandle().pMinimized);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, std::shared_ptr<FlagTool>& o) {
  if (!o) {
    o = std::make_shared<FlagTool>(sInputManager, sSolarSystem, sSettings, "");
  }

  std::string object;
  cs::core::Settings::deserialize(j, "object", object);
  o->setObjectName(object);

  cs::core::Settings::deserialize(j, "lngLat", o->pLngLat);
  cs::core::Settings::deserialize(j, "color", o->pColor);
  cs::core::Settings::deserialize(j, "scaleDistance", o->pScaleDistance);
  cs::core::Settings::deserialize(j, "text", o->pText);
  cs::core::Settings::deserialize(j, "minimized", o->pMinimized);
}

void to_json(nlohmann::json& j, std::shared_ptr<FlagTool> const& o) {
  cs::core::Settings::serialize(j, "object", o->getObjectName());
  cs::core::Settings::serialize(j, "lngLat", o->pLngLat);
  cs::core::Settings::serialize(j, "color", o->pColor);
  cs::core::Settings::serialize(j, "scaleDistance", o->pScaleDistance);
  cs::core::Settings::serialize(j, "text", o->pText);
  cs::core::Settings::serialize(j, "minimized", o->pMinimized);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, std::shared_ptr<PathTool>& o) {
  if (!o) {
    o = std::make_shared<PathTool>(sInputManager, sSolarSystem, sSettings, "");
  }

  std::string object;
  cs::core::Settings::deserialize(j, "object", object);
  o->setObjectName(object);

  cs::core::Settings::deserialize(j, "color", o->pColor);
  cs::core::Settings::deserialize(j, "scaleDistance", o->pScaleDistance);
  cs::core::Settings::deserialize(j, "text", o->pText);

  std::vector<glm::dvec2> positions;
  cs::core::Settings::deserialize(j, "positions", positions);
  o->setPositions(positions);
}

void to_json(nlohmann::json& j, std::shared_ptr<PathTool> const& o) {
  cs::core::Settings::serialize(j, "object", o->getObjectName());
  cs::core::Settings::serialize(j, "color", o->pColor);
  cs::core::Settings::serialize(j, "scaleDistance", o->pScaleDistance);
  cs::core::Settings::serialize(j, "text", o->pText);
  cs::core::Settings::serialize(j, "positions", o->getPositions());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, std::shared_ptr<PolygonTool>& o) {
  if (!o) {
    o = std::make_shared<PolygonTool>(sInputManager, sSolarSystem, sSettings, "");
  }

  std::string object;
  cs::core::Settings::deserialize(j, "object", object);
  o->setObjectName(object);

  cs::core::Settings::deserialize(j, "color", o->pColor);
  cs::core::Settings::deserialize(j, "scaleDistance", o->pScaleDistance);
  cs::core::Settings::deserialize(j, "showMesh", o->pShowMesh);
  cs::core::Settings::deserialize(j, "text", o->pText);

  std::vector<glm::dvec2> positions;
  cs::core::Settings::deserialize(j, "positions", positions);
  o->setPositions(positions);
}

void to_json(nlohmann::json& j, std::shared_ptr<PolygonTool> const& o) {
  cs::core::Settings::serialize(j, "object", o->getObjectName());
  cs::core::Settings::serialize(j, "color", o->pColor);
  cs::core::Settings::serialize(j, "scaleDistance", o->pScaleDistance);
  cs::core::Settings::serialize(j, "showMesh", o->pShowMesh);
  cs::core::Settings::serialize(j, "text", o->pText);
  cs::core::Settings::serialize(j, "positions", o->getPositions());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  deserializeTools(j, "dipStrikes", o.mDipStrikes);
  deserializeTools(j, "ellipses", o.mEllipses);
  deserializeTools(j, "flags", o.mFlags);
  deserializeTools(j, "paths", o.mPaths);
  deserializeTools(j, "polygons", o.mPolygons);

  cs::core::Settings::deserialize(j, "polygonHeightDiff", o.mPolygonHeightDiff);
  cs::core::Settings::deserialize(j, "polygonMaxAttempt", o.mPolygonMaxAttempt);
  cs::core::Settings::deserialize(j, "polygonMaxPoints", o.mPolygonMaxPoints);
  cs::core::Settings::deserialize(j, "polygonSleekness", o.mPolygonSleekness);
  cs::core::Settings::deserialize(j, "ellipseSamples", o.mEllipseSamples);
  cs::core::Settings::deserialize(j, "pathSamples", o.mPathSamples);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "dipStrikes", o.mDipStrikes);
  cs::core::Settings::serialize(j, "ellipses", o.mEllipses);
  cs::core::Settings::serialize(j, "flags", o.mFlags);
  cs::core::Settings::serialize(j, "paths", o.mPaths);
  cs::core::Settings::serialize(j, "polygons", o.mPolygons);
  cs::core::Settings::serialize(j, "polygonHeightDiff", o.mPolygonHeightDiff);
  cs::core::Settings::serialize(j, "polygonMaxAttempt", o.mPolygonMaxAttempt);
  cs::core::Settings::serialize(j, "polygonMaxPoints", o.mPolygonMaxPoints);
  cs::core::Settings::serialize(j, "polygonSleekness", o.mPolygonSleekness);
  cs::core::Settings::serialize(j, "ellipseSamples", o.mEllipseSamples);
  cs::core::Settings::serialize(j, "pathSamples", o.mPathSamples);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  mGuiManager->addPluginTabToSideBarFromHTML(
      "Measurement Tools", "multiline_chart", "../share/resources/gui/measurement-tools-tab.html");

  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-measurement-tools.js");
  mGuiManager->addCSS("css/csp-measurement-tools-sidebar.css");

  mGuiManager->getGui()->callJavascript(
      "CosmoScout.measurementTools.add", "Location Flag", "edit_location");
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.measurementTools.add", "Landing Ellipse", "location_searching");
  mGuiManager->getGui()->callJavascript("CosmoScout.measurementTools.add", "Path", "timeline");
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.measurementTools.add", "Dip & Strike", "clear_all");
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.measurementTools.add", "Polygon", "crop_landscape");

  mGuiManager->getGui()->registerCallback("measurementTools.setNext",
      "Selects which tool will be created next. The given string should be either 'Location Flag', "
      "'Landing Ellipse, 'Path', 'Dip & Strike' or 'Polygon'.",
      std::function([this](std::string&& name) { mNextTool = name; }));

  mOnClickConnection = mInputManager->pButtons[0].connect([this](bool pressed) {
    if (!pressed && !mInputManager->pHoveredGuiItem.get()) {
      auto object     = mInputManager->pHoveredObject.get().mObject;
      auto objectName = mInputManager->pHoveredObject.get().mObjectName;

      if (!object) {
        return;
      }

      auto radii = object->getRadii();
      if (mNextTool == "Location Flag") {
        auto tool =
            std::make_shared<FlagTool>(mInputManager, mSolarSystem, mAllSettings, objectName);
        tool->pLngLat = cs::utils::convert::cartesianToLngLat(
            mInputManager->pHoveredObject.get().mPosition, radii);
        mPluginSettings.mFlags.push_back(tool);

      } else if (mNextTool == "Landing Ellipse") {
        auto tool =
            std::make_shared<EllipseTool>(mInputManager, mSolarSystem, mAllSettings, objectName);
        tool->getCenterHandle().pLngLat = cs::utils::convert::cartesianToLngLat(
            mInputManager->pHoveredObject.get().mPosition, radii);
        tool->setNumSamples(mPluginSettings.mEllipseSamples.get());
        mPluginSettings.mEllipses.push_back(tool);

      } else if (mNextTool == "Path") {
        auto tool =
            std::make_shared<PathTool>(mInputManager, mSolarSystem, mAllSettings, objectName);
        tool->setNumSamples(mPluginSettings.mPathSamples.get());
        tool->pAddPointMode = true;
        tool->addPoint();
        mPluginSettings.mPaths.push_back(tool);

      } else if (mNextTool == "Dip & Strike") {
        auto tool =
            std::make_shared<DipStrikeTool>(mInputManager, mSolarSystem, mAllSettings, objectName);
        tool->pAddPointMode = true;
        tool->addPoint();
        mPluginSettings.mDipStrikes.push_back(tool);

      } else if (mNextTool == "Polygon") {
        auto tool =
            std::make_shared<PolygonTool>(mInputManager, mSolarSystem, mAllSettings, objectName);
        tool->setHeightDiff(mPluginSettings.mPolygonHeightDiff.get());
        tool->setMaxAttempt(mPluginSettings.mPolygonMaxAttempt.get());
        tool->setMaxPoints(mPluginSettings.mPolygonMaxPoints.get());
        tool->setSleekness(mPluginSettings.mPolygonSleekness.get());
        tool->pAddPointMode = true;
        tool->addPoint();
        mPluginSettings.mPolygons.push_back(tool);

      } else if (mNextTool != "none") {
        logger().warn("Failed to create tool '{}': This is an unknown tool type!", mNextTool);
      }

      mNextTool = "none";
      mGuiManager->getGui()->callJavascript("CosmoScout.measurementTools.deselect");
    }
  });

  mOnDoubleClickConnection = mInputManager->sOnDoubleClick.connect([this]() {
    mNextTool = "none";
    mGuiManager->getGui()->callJavascript("CosmoScout.measurementTools.deselect");
  });

  mPluginSettings.mPolygonHeightDiff.connect([this](float val) {
    for (auto& p : mPluginSettings.mPolygons) {
      p->setHeightDiff(val);
    }
  });

  mPluginSettings.mPolygonMaxAttempt.connect([this](int32_t val) {
    for (auto& p : mPluginSettings.mPolygons) {
      p->setMaxAttempt(val);
    }
  });

  mPluginSettings.mPolygonMaxPoints.connect([this](int32_t val) {
    for (auto& p : mPluginSettings.mPolygons) {
      p->setMaxPoints(val);
    }
  });

  mPluginSettings.mPolygonSleekness.connect([this](int32_t val) {
    for (auto& p : mPluginSettings.mPolygons) {
      p->setSleekness(val);
    }
  });

  mPluginSettings.mEllipseSamples.connect([this](int32_t val) {
    for (auto& p : mPluginSettings.mEllipses) {
      p->setNumSamples(val);
    }
  });

  mPluginSettings.mPathSamples.connect([this](int32_t val) {
    for (auto& p : mPluginSettings.mPaths) {
      p->setNumSamples(val);
    }
  });

  // Load settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Save settings as this plugin may get reloaded.
  onSave();

  mGuiManager->removePluginTab("Measurement Tools");

  mGuiManager->getGui()->unregisterCallback("measurementTools.setNext");
  mGuiManager->removeCSS("css/csp-measurement-tools-sidebar.css");

  mInputManager->pButtons[0].disconnect(mOnClickConnection);
  mInputManager->sOnDoubleClick.disconnect(mOnDoubleClickConnection);

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  // Update all registered tools. If the pShouldDelete property is set, the Tool is removed from the
  // list.
  updateTools(mPluginSettings.mDipStrikes);
  updateTools(mPluginSettings.mEllipses);
  updateTools(mPluginSettings.mFlags);
  updateTools(mPluginSettings.mPaths);
  updateTools(mPluginSettings.mPolygons);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  sInputManager = mInputManager;
  sSolarSystem  = mSolarSystem;
  sSettings     = mAllSettings;
  sTimeControl  = mTimeControl;

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-measurement-tools"), mPluginSettings);

  sInputManager.reset();
  sSolarSystem.reset();
  sSettings.reset();
  sTimeControl.reset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-measurement-tools"] = mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::measurementtools
