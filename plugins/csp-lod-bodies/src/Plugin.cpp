////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "LodBody.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::lodbodies::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Dataset& o) {
  cs::core::Settings::deserialize(j, "copyright", o.mCopyright);
  cs::core::Settings::deserialize(j, "layers", o.mLayers);
  cs::core::Settings::deserialize(j, "maxLevel", o.mMaxLevel);
  cs::core::Settings::deserialize(j, "url", o.mURL);
}

void to_json(nlohmann::json& j, Plugin::Settings::Dataset const& o) {
  cs::core::Settings::serialize(j, "copyright", o.mCopyright);
  cs::core::Settings::serialize(j, "layers", o.mLayers);
  cs::core::Settings::serialize(j, "maxLevel", o.mMaxLevel);
  cs::core::Settings::serialize(j, "url", o.mURL);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Body& o) {
  cs::core::Settings::deserialize(j, "activeDemDataset", o.mActiveDemDataset);
  cs::core::Settings::deserialize(j, "activeImgDataset", o.mActiveImgDataset);
  cs::core::Settings::deserialize(j, "demDatasets", o.mDemDatasets);
  cs::core::Settings::deserialize(j, "imgDatasets", o.mImgDatasets);
  cs::core::Settings::deserialize(j, "brdfHdr", o.mBrdfHdr);
  cs::core::Settings::deserialize(j, "brdfNonHdr", o.mBrdfNonHdr);
  cs::core::Settings::deserialize(j, "avgLinearImgIntensity", o.mAvgLinearImgIntensity);
}

void to_json(nlohmann::json& j, Plugin::Settings::Body const& o) {
  cs::core::Settings::serialize(j, "activeDemDataset", o.mActiveDemDataset);
  cs::core::Settings::serialize(j, "activeImgDataset", o.mActiveImgDataset);
  cs::core::Settings::serialize(j, "demDatasets", o.mDemDatasets);
  cs::core::Settings::serialize(j, "imgDatasets", o.mImgDatasets);
  cs::core::Settings::serialize(j, "brdfHdr", o.mBrdfHdr);
  cs::core::Settings::serialize(j, "brdfNonHdr", o.mBrdfNonHdr);
  cs::core::Settings::serialize(j, "avgLinearImgIntensity", o.mAvgLinearImgIntensity);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::BRDF& o) {
  cs::core::Settings::deserialize(j, "source", o.source);
  cs::core::Settings::deserialize(j, "properties", o.properties);
}

void to_json(nlohmann::json& j, Plugin::Settings::BRDF const& o) {
  cs::core::Settings::serialize(j, "source", o.source);
  cs::core::Settings::serialize(j, "properties", o.properties);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "terrainProjectionType", o.mTerrainProjectionType);
  cs::core::Settings::deserialize(j, "lodFactor", o.mLODFactor);
  cs::core::Settings::deserialize(j, "autoLod", o.mAutoLOD);
  cs::core::Settings::deserialize(j, "autoLodRange", o.mAutoLODRange);
  cs::core::Settings::deserialize(j, "autoLodFrameTimeRange", o.mAutoLODFrameTimeRange);
  cs::core::Settings::deserialize(j, "textureGamma", o.mTextureGamma);
  cs::core::Settings::deserialize(j, "enableHeightlines", o.mEnableHeightlines);
  cs::core::Settings::deserialize(j, "enableLatLongGrid", o.mEnableLatLongGrid);
  cs::core::Settings::deserialize(j, "colorMappingType", o.mColorMappingType);
  cs::core::Settings::deserialize(j, "terrainColorMap", o.mTerrainColorMap);
  cs::core::Settings::deserialize(j, "heightRange", o.mHeightRange);
  cs::core::Settings::deserialize(j, "slopeRange", o.mSlopeRange);
  cs::core::Settings::deserialize(j, "enableWireframe", o.mEnableWireframe);
  cs::core::Settings::deserialize(j, "enableBounds", o.mEnableBounds);
  cs::core::Settings::deserialize(j, "enableTilesDebug", o.mEnableTilesDebug);
  cs::core::Settings::deserialize(j, "enableTilesFreeze", o.mEnableTilesFreeze);
  cs::core::Settings::deserialize(j, "maxGPUTilesColor", o.mMaxGPUTilesColor);
  cs::core::Settings::deserialize(j, "maxGPUTilesDEM", o.mMaxGPUTilesDEM);
  cs::core::Settings::deserialize(j, "tileResolutionDEM", o.mTileResolutionDEM);
  cs::core::Settings::deserialize(j, "tileResolutionIMG", o.mTileResolutionIMG);
  cs::core::Settings::deserialize(j, "mapCache", o.mMapCache);
  cs::core::Settings::deserialize(j, "bodies", o.mBodies);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "terrainProjectionType", o.mTerrainProjectionType);
  cs::core::Settings::serialize(j, "lodFactor", o.mLODFactor);
  cs::core::Settings::serialize(j, "autoLod", o.mAutoLOD);
  cs::core::Settings::serialize(j, "autoLodRange", o.mAutoLODRange);
  cs::core::Settings::serialize(j, "autoLodFrameTimeRange", o.mAutoLODFrameTimeRange);
  cs::core::Settings::serialize(j, "textureGamma", o.mTextureGamma);
  cs::core::Settings::serialize(j, "enableHeightlines", o.mEnableHeightlines);
  cs::core::Settings::serialize(j, "enableLatLongGrid", o.mEnableLatLongGrid);
  cs::core::Settings::serialize(j, "colorMappingType", o.mColorMappingType);
  cs::core::Settings::serialize(j, "terrainColorMap", o.mTerrainColorMap);
  cs::core::Settings::serialize(j, "heightRange", o.mHeightRange);
  cs::core::Settings::serialize(j, "slopeRange", o.mSlopeRange);
  cs::core::Settings::serialize(j, "enableWireframe", o.mEnableWireframe);
  cs::core::Settings::serialize(j, "enableBounds", o.mEnableBounds);
  cs::core::Settings::serialize(j, "enableTilesDebug", o.mEnableTilesDebug);
  cs::core::Settings::serialize(j, "enableTilesFreeze", o.mEnableTilesFreeze);
  cs::core::Settings::serialize(j, "maxGPUTilesColor", o.mMaxGPUTilesColor);
  cs::core::Settings::serialize(j, "maxGPUTilesDEM", o.mMaxGPUTilesDEM);
  cs::core::Settings::serialize(j, "tileResolutionDEM", o.mTileResolutionDEM);
  cs::core::Settings::serialize(j, "tileResolutionIMG", o.mTileResolutionIMG);
  cs::core::Settings::serialize(j, "mapCache", o.mMapCache);
  cs::core::Settings::serialize(j, "bodies", o.mBodies);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  mGuiManager->addPluginTabToSideBarFromHTML(
      "Body Settings", "landscape", "../share/resources/gui/lod_body_tab.html");
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Body Settings", "landscape", "../share/resources/gui/lod_body_settings.html");
  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-lod-bodies.js");

  mGuiManager->getGui()->registerCallback("lodBodies.setEnableTilesFreeze",
      "If set to true, the level of detail and the frustum culling of the planet's tiles will not "
      "be updated anymore.",
      std::function([this](bool enable) { mPluginSettings->mEnableTilesFreeze = enable; }));
  mPluginSettings->mEnableTilesFreeze.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("lodBodies.setEnableTilesFreeze", enable);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setEnableTilesDebug",
      "Enables or disables debug coloring of the planet's tiles.",
      std::function([this](bool enable) { mPluginSettings->mEnableTilesDebug = enable; }));
  mPluginSettings->mEnableTilesDebug.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("lodBodies.setEnableTilesDebug", enable);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setEnableWireframe",
      "Enables or disables wireframe rendering of the planet.",
      std::function([this](bool enable) { mPluginSettings->mEnableWireframe = enable; }));
  mPluginSettings->mEnableWireframe.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("lodBodies.setEnableWireframe", enable);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setEnableBounds",
      "Enables or disables bounding box rendering of the planet's tiles.",
      std::function([this](bool enable) { mPluginSettings->mEnableBounds = enable; }));
  mPluginSettings->mEnableBounds.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("lodBodies.setEnableBounds", enable); });

  mGuiManager->getGui()->registerCallback("lodBodies.setEnableHeightlines",
      "Enables or disables rendering of iso-altitude lines.",
      std::function([this](bool enable) { mPluginSettings->mEnableHeightlines = enable; }));
  mPluginSettings->mEnableHeightlines.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("lodBodies.setEnableHeightlines", enable);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setEnableLatLongGrid",
      "Enables or disables rendering of a latidude-longitude-grid.",
      std::function([this](bool enable) { mPluginSettings->mEnableLatLongGrid = enable; }));
  mPluginSettings->mEnableLatLongGrid.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("lodBodies.setEnableLatLongGrid", enable);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setTerrainLod",
      "Specifies the amount of detail of the planet's surface. Should be in the range 1-100.",
      std::function(
          [this](double value) { mPluginSettings->mLODFactor = static_cast<float>(value); }));
  mPluginSettings->mLODFactor.connectAndTouch([this](float value) {
    for (auto&& body : mLodBodies) {
      body.second->setLODFactor(value);
    }
    mGuiManager->setSliderValue("lodBodies.setTerrainLod", value);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setEnableAutoTerrainLod",
      "If set to true, the level-of-detail will be chosen automatically based on the current "
      "rendering performance.",
      std::function([this](bool enable) { mPluginSettings->mAutoLOD = enable; }));
  mPluginSettings->mAutoLOD.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("lodBodies.setEnableAutoTerrainLod", enable);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setAutoLoDRange",
      "Sets the minimum and maximum LoD value for auto-level-of-detail.",
      std::function([this](double val1, double val2) {
        mPluginSettings->mAutoLODRange = glm::vec2(val1, val2);
      }));
  mPluginSettings->mAutoLODRange.connectAndTouch([this](glm::vec2 const& val) {
    mGuiManager->setSliderValue("lodBodies.setAutoLoDRange", val);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setTextureGamma",
      "A multiplier for the brightness of the image channel.", std::function([this](double value) {
        mPluginSettings->mTextureGamma = static_cast<float>(value);
      }));
  mPluginSettings->mTextureGamma.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("lodBodies.setTextureGamma", value); });

  mGuiManager->getGui()->registerCallback("lodBodies.setHeightRange",
      "Sets the height range for the color mapping in kilometers.",
      std::function([this](double val1, double val2) {
        mPluginSettings->mHeightRange = glm::vec2(val1, val2);
      }));
  mPluginSettings->mHeightRange.connectAndTouch([this](glm::vec2 const& value) {
    mGuiManager->setSliderValue("lodBodies.setHeightRange", value);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setSlopeRange",
      "Sets the slope range for the color mapping in degrees.",
      std::function([this](double val1, double val2) {
        mPluginSettings->mSlopeRange = glm::vec2(val1, val2);
      }));
  mPluginSettings->mSlopeRange.connectAndTouch([this](glm::vec2 const& value) {
    mGuiManager->setSliderValue("lodBodies.setSlopeRange", value);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setSurfaceColoringMode0",
      "Call this to deselect any surface coloring.", std::function([this] {
        mPluginSettings->mColorMappingType = Settings::ColorMappingType::eNone;
      }));
  mGuiManager->getGui()->registerCallback("lodBodies.setSurfaceColoringMode1",
      "Call this to enable height based surface coloring.", std::function([this] {
        mPluginSettings->mColorMappingType = Settings::ColorMappingType::eHeight;
      }));
  mGuiManager->getGui()->registerCallback("lodBodies.setSurfaceColoringMode2",
      "Call this to enable slope based surface coloring.", std::function([this] {
        mPluginSettings->mColorMappingType = Settings::ColorMappingType::eSlope;
      }));
  mPluginSettings->mColorMappingType.connect([this](Settings::ColorMappingType type) {
    if (type == Settings::ColorMappingType::eNone) {
      mGuiManager->setRadioChecked("lodBodies.setSurfaceColoringMode0");
    } else if (type == Settings::ColorMappingType::eHeight) {
      mGuiManager->setRadioChecked("lodBodies.setSurfaceColoringMode1");
    } else if (type == Settings::ColorMappingType::eSlope) {
      mGuiManager->setRadioChecked("lodBodies.setSurfaceColoringMode2");
    }
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setTerrainProjectionMode0",
      "Call this to use a GPU-based HEALPix projection for the planet's surface.",
      std::function([this] {
        mPluginSettings->mTerrainProjectionType = Settings::TerrainProjectionType::eHEALPix;
      }));
  mGuiManager->getGui()->registerCallback("lodBodies.setTerrainProjectionMode1",
      "Call this to use a CPU-based HEALPix projection and a linear interpolation on the GPU-side "
      "for the planet's surface.",
      std::function([this] {
        mPluginSettings->mTerrainProjectionType = Settings::TerrainProjectionType::eLinear;
      }));
  mGuiManager->getGui()->registerCallback("lodBodies.setTerrainProjectionMode2",
      "Call this to choose a projection for the planet's surface based on the observer's distance.",
      std::function([this] {
        mPluginSettings->mTerrainProjectionType = Settings::TerrainProjectionType::eHybrid;
      }));
  mPluginSettings->mTerrainProjectionType.connect([this](Settings::TerrainProjectionType type) {
    if (type == Settings::TerrainProjectionType::eHEALPix) {
      mGuiManager->setRadioChecked("lodBodies.setTerrainProjectionMode0");
    } else if (type == Settings::TerrainProjectionType::eLinear) {
      mGuiManager->setRadioChecked("lodBodies.setTerrainProjectionMode1");
    } else if (type == Settings::TerrainProjectionType::eHybrid) {
      mGuiManager->setRadioChecked("lodBodies.setTerrainProjectionMode2");
    }
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setTilesImg",
      "Set the current planet's image channel to the TileSource with the given name.",
      std::function([this](std::string&& name) {
        if (mSolarSystem->pActiveObject.get()) {
          auto body =
              std::dynamic_pointer_cast<LodBody>(mSolarSystem->pActiveObject.get()->getSurface());

          if (body) {
            setImageSource(body, name);
          }
        }
      }));

  mGuiManager->getGui()->registerCallback("lodBodies.setTilesDem",
      "Set the current planet's elevation channel to the TileSource with the given name.",
      std::function([this](std::string&& name) {
        if (mSolarSystem->pActiveObject.get()) {
          auto body =
              std::dynamic_pointer_cast<LodBody>(mSolarSystem->pActiveObject.get()->getSurface());

          if (body) {
            setElevationSource(body, name);
          }
        }
      }));

  mActiveObjectConnection = mSolarSystem->pActiveObject.connectAndTouch(
      [this](std::shared_ptr<const cs::scene::CelestialObject> const& body) {
        bool tabEnabled = false;

        if (body) {
          auto lodBody = std::dynamic_pointer_cast<LodBody>(body->getSurface());

          if (lodBody) {

            tabEnabled = true;

            mGuiManager->getGui()->callJavascript(
                "CosmoScout.gui.clearDropdown", "lodBodies.setTilesImg");
            mGuiManager->getGui()->callJavascript(
                "CosmoScout.gui.clearDropdown", "lodBodies.setTilesDem");
            mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue",
                "lodBodies.setTilesImg", "None", "None", "false");

            auto const& settings = getBodySettings(lodBody);
            for (auto const& source : settings.mImgDatasets) {
              bool active = source.first == settings.mActiveImgDataset;
              mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue",
                  "lodBodies.setTilesImg", source.first, source.first, active);
              if (active) {
                mGuiManager->getGui()->callJavascript(
                    "CosmoScout.lodBodies.setMapDataCopyright", source.second.mCopyright);
              }
            }

            for (auto const& source : settings.mDemDatasets) {
              bool active = source.first == settings.mActiveDemDataset;
              mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue",
                  "lodBodies.setTilesDem", source.first, source.first, active);
              if (active) {
                mGuiManager->getGui()->callJavascript(
                    "CosmoScout.lodBodies.setElevationDataCopyright", source.second.mCopyright);
              }
            }
          }
        }

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.sidebar.setTabEnabled", "Body Settings", tabEnabled);
      });

  mAutoLod = mPluginSettings->mLODFactor.get();

  // If auto-LoD gets enabled, the auto-LoD factor gets initialized with the current manual LoD
  // factor. If it is disabled, we reset the slider position to the manual LoD.
  mPluginSettings->mAutoLOD.connect([this](bool enabled) {
    if (enabled) {
      mAutoLod = mPluginSettings->mLODFactor.get();
    } else {
      mGuiManager->setSliderValue("lodBodies.setTerrainLod", mPluginSettings->mLODFactor.get());
    }
  });

  mPluginSettings->mMapCache.connect([this](std::string const& val) {
    for (auto&& body : mLodBodies) {
      auto src =
          std::dynamic_pointer_cast<TileSourceWebMapService>(body.second->getDEMtileSource());
      if (src) {
        src->setCacheDirectory(val);
      }
      src = std::dynamic_pointer_cast<TileSourceWebMapService>(body.second->getIMGtileSource());
      if (src) {
        src->setCacheDirectory(val);
      }
    }
  });

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Save settings as this plugin may get reloaded.
  onSave();

  mSolarSystem->pActiveObject.disconnect(mActiveObjectConnection);

  for (auto const& [name, body] : mLodBodies) {
    unregisterBody(name);
  }

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  mGuiManager->removePluginTab("Body Settings");
  mGuiManager->removeSettingsSection("Body Settings");

  mGuiManager->getGui()->callJavascript("CosmoScout.removeApi", "lodBodies");

  mGuiManager->getGui()->unregisterCallback("lodBodies.setAutoLoDRange");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableTilesFreeze");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableTilesDebug");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableWireframe");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableBounds");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableHeightlines");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableLatLongGrid");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setTerrainLod");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableAutoTerrainLod");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setTextureGamma");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setHeightRange");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setSlopeRange");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setSurfaceColoringMode0");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setSurfaceColoringMode1");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setSurfaceColoringMode2");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setTerrainProjectionMode0");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setTerrainProjectionMode1");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setTerrainProjectionMode2");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setTilesImg");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setTilesDem");

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  if (mPluginSettings->mAutoLOD.get()) {

    double minLODFactor = mPluginSettings->mAutoLODRange.get().x;
    double maxLODFactor = mPluginSettings->mAutoLODRange.get().y;

    // These numbers shall ensure that the frame rate stays above 60 Hz (16.6ms). Somehow we should
    // try to retrieve the actual refresh rate of the display in the future.
    double minTime = mPluginSettings->mAutoLODFrameTimeRange.get().x;
    double maxTime = mPluginSettings->mAutoLODFrameTimeRange.get().y;

    if (cs::utils::FrameStats::get().pFrameTime.get() > maxTime) {
      mAutoLod = static_cast<float>(std::max(minLODFactor,
          mAutoLod -
              std::min(1.0, 0.1 * (cs::utils::FrameStats::get().pFrameTime.get() - maxTime))));
    } else if (cs::utils::FrameStats::get().pFrameTime.get() < minTime) {
      mAutoLod = static_cast<float>(std::min(maxLODFactor,
          mAutoLod +
              std::min(1.0, 0.02 * (minTime - cs::utils::FrameStats::get().pFrameTime.get()))));
    }

    // Apply the computed LoD factor.
    for (auto&& body : mLodBodies) {
      body.second->setLODFactor(mAutoLod);
    }

    mGuiManager->setSliderValue("lodBodies.setTerrainLod", mAutoLod);
  }

  for (auto const& [name, body] : mLodBodies) {
    body->update();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-lod-bodies"), *mPluginSettings);

  // For now, we cannot re-create the GLResources.
  if (!mGLResources) {
    mGLResources = std::make_shared<csp::lodbodies::GLResources>(
        mPluginSettings->mMaxGPUTilesDEM.get(), mPluginSettings->mMaxGPUTilesColor.get(),
        mPluginSettings->mTileResolutionDEM.get(), mPluginSettings->mTileResolutionIMG.get());

    mPluginSettings->mMaxGPUTilesColor.connect([](uint32_t /*val*/) {
      logger().warn("Changing the maximum number of allocated color tiles at run-time is not "
                    "supported. Please restart CosmoScout VR!");
    });

    mPluginSettings->mMaxGPUTilesDEM.connect([](uint32_t /*val*/) {
      logger().warn("Changing the maximum number of allocated elevation tiles at run-time is not "
                    "supported. Please restart CosmoScout VR!");
    });

    mPluginSettings->mTileResolutionDEM.connect([](uint32_t /*val*/) {
      logger().warn("Changing the tile resolution at run-time is not supported. Please restart "
                    "CosmoScout VR!");
    });

    mPluginSettings->mTileResolutionIMG.connect([](uint32_t /*val*/) {
      logger().warn("Changing the tile resolution at run-time is not supported. Please restart "
                    "CosmoScout VR!");
    });
  }

  // First try to re-configure existing lodBodies. We assume that they are similar if they have
  // the same name in the settings (which means they are attached to an anchor with the same name).
  auto lodBody = mLodBodies.begin();
  while (lodBody != mLodBodies.end()) {
    auto settings = mPluginSettings->mBodies.find(lodBody->first);
    // If there are settings for this lodBody, reconfigure it.
    if (settings != mPluginSettings->mBodies.end()) {
      lodBody->second->setObjectName(settings->first);

      setImageSource(lodBody->second, settings->second.mActiveImgDataset);
      setElevationSource(lodBody->second, settings->second.mActiveDemDataset);

      ++lodBody;
    } else {
      // Else delete it.
      unregisterBody(lodBody->first);
      lodBody = mLodBodies.erase(lodBody);
    }
  }

  // Then add new lodBodies.
  for (auto const& settings : mPluginSettings->mBodies) {

    // Skip already existing bodies.
    if (mLodBodies.find(settings.first) != mLodBodies.end()) {
      continue;
    }

    auto body = std::make_shared<LodBody>(
        mAllSettings, mGraphicsEngine, mSolarSystem, mPluginSettings, mGuiManager, mGLResources);

    body->setObjectName(settings.first);

    auto object = mSolarSystem->getObject(settings.first);
    object->setSurface(body);
    object->setIntersectableObject(body);

    mLodBodies.emplace(settings.first, body);

    setImageSource(body, settings.second.mActiveImgDataset);
    setElevationSource(body, settings.second.mActiveDemDataset);
  }

  mSolarSystem->pActiveObject.touch(mActiveObjectConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-lod-bodies"] = *mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::unregisterBody(std::string const& name) {
  auto object = mSolarSystem->getObject(name);
  object->setSurface(nullptr);
  object->setIntersectableObject(nullptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Plugin::Settings::Body& Plugin::getBodySettings(std::shared_ptr<LodBody> const& body) const {
  auto name = std::find_if(
      mLodBodies.begin(), mLodBodies.end(), [&](auto const& pair) { return pair.second == body; });
  return mPluginSettings->mBodies.at(name->first);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setImageSource(std::shared_ptr<LodBody> const& body, std::string const& name) const {
  auto& settings = getBodySettings(body);

  if (name == "None") {
    body->setIMGtileSource(nullptr, 0);
    mGuiManager->getGui()->callJavascript("CosmoScout.lodBodies.setMapDataCopyright", "");
    settings.mActiveImgDataset = "None";
  } else {
    auto dataset = settings.mImgDatasets.find(name);
    if (dataset == settings.mImgDatasets.end()) {
      logger().warn("Cannot set image dataset '{}': There is no dataset defined with this name! "
                    "Using first dataset instead...",
          name);
      dataset = settings.mImgDatasets.begin();
    }

    settings.mActiveImgDataset = dataset->first;

    auto source =
        std::make_shared<TileSourceWebMapService>(mPluginSettings->mTileResolutionIMG.get());
    source->setCacheDirectory(mPluginSettings->mMapCache.get());
    source->setLayers(dataset->second.mLayers);
    source->setUrl(dataset->second.mURL);
    source->setDataType(TileDataType::eColor);

    body->setIMGtileSource(source, dataset->second.mMaxLevel);

    mGuiManager->getGui()->callJavascript(
        "CosmoScout.lodBodies.setMapDataCopyright", dataset->second.mCopyright);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setElevationSource(
    std::shared_ptr<LodBody> const& body, std::string const& name) const {

  auto& settings = getBodySettings(body);
  auto  dataset  = settings.mDemDatasets.find(name);
  if (dataset == settings.mDemDatasets.end()) {
    logger().warn("Cannot set elevation dataset '{}': There is no dataset defined with this name! "
                  "Using first dataset instead...",
        name);
    dataset = settings.mDemDatasets.begin();
  }

  settings.mActiveDemDataset = dataset->first;

  auto source =
      std::make_shared<TileSourceWebMapService>(mPluginSettings->mTileResolutionDEM.get());
  source->setCacheDirectory(mPluginSettings->mMapCache.get());
  source->setLayers(dataset->second.mLayers);
  source->setUrl(dataset->second.mURL);
  source->setDataType(TileDataType::eElevation);

  body->setDEMtileSource(source, dataset->second.mMaxLevel);

  mGuiManager->getGui()->callJavascript(
      "CosmoScout.lodBodies.setElevationDataCopyright", dataset->second.mCopyright);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
