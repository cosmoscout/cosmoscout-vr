////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "LodBody.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
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

void from_json(nlohmann::json const& j, TileDataType& o) {
  auto s = j.get<std::string>();
  if (s == "Float32") {
    o = TileDataType::eFloat32;
  } else if (s == "UInt8") {
    o = TileDataType::eUInt8;
  } else if (s == "U8Vec3") {
    o = TileDataType::eU8Vec3;
  } else {
    throw std::runtime_error(
        "Failed to parse TileDataType! Only 'Float32', 'UInt8' or 'U8Vec3' are allowed.");
  }
}

void to_json(nlohmann::json& j, TileDataType o) {
  switch (o) {
  case TileDataType::eFloat32:
    j = "Float32";
    break;
  case TileDataType::eUInt8:
    j = "UInt8";
    break;
  case TileDataType::eU8Vec3:
    j = "U8Vec3";
    break;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Dataset& o) {
  cs::core::Settings::deserialize(j, "format", o.mFormat);
  cs::core::Settings::deserialize(j, "copyright", o.mCopyright);
  cs::core::Settings::deserialize(j, "layers", o.mLayers);
  cs::core::Settings::deserialize(j, "maxLevel", o.mMaxLevel);
  cs::core::Settings::deserialize(j, "url", o.mURL);
}

void to_json(nlohmann::json& j, Plugin::Settings::Dataset const& o) {
  cs::core::Settings::serialize(j, "format", o.mFormat);
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
}

void to_json(nlohmann::json& j, Plugin::Settings::Body const& o) {
  cs::core::Settings::serialize(j, "activeDemDataset", o.mActiveDemDataset);
  cs::core::Settings::serialize(j, "activeImgDataset", o.mActiveImgDataset);
  cs::core::Settings::serialize(j, "demDatasets", o.mDemDatasets);
  cs::core::Settings::serialize(j, "imgDatasets", o.mImgDatasets);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "terrainProjectionType", o.mTerrainProjectionType);
  cs::core::Settings::deserialize(j, "lodFactor", o.mLODFactor);
  cs::core::Settings::deserialize(j, "autoLod", o.mAutoLOD);
  cs::core::Settings::deserialize(j, "textureGamma", o.mTextureGamma);
  cs::core::Settings::deserialize(j, "enableHeightlines", o.mEnableHeightlines);
  cs::core::Settings::deserialize(j, "enableLatLongGrid", o.mEnableLatLongGrid);
  cs::core::Settings::deserialize(j, "colorMappingType", o.mColorMappingType);
  cs::core::Settings::deserialize(j, "terrainColorMap", o.mTerrainColorMap);
  cs::core::Settings::deserialize(j, "enableColorMixing", o.mEnableColorMixing);
  cs::core::Settings::deserialize(j, "heightRange", o.mHeightRange);
  cs::core::Settings::deserialize(j, "slopeRange", o.mSlopeRange);
  cs::core::Settings::deserialize(j, "enableWireframe", o.mEnableWireframe);
  cs::core::Settings::deserialize(j, "enableTilesDebug", o.mEnableTilesDebug);
  cs::core::Settings::deserialize(j, "enableTilesFreeze", o.mEnableTilesFreeze);
  cs::core::Settings::deserialize(j, "maxGPUTilesColor", o.mMaxGPUTilesColor);
  cs::core::Settings::deserialize(j, "maxGPUTilesGray", o.mMaxGPUTilesGray);
  cs::core::Settings::deserialize(j, "maxGPUTilesDEM", o.mMaxGPUTilesDEM);
  cs::core::Settings::deserialize(j, "mapCache", o.mMapCache);
  cs::core::Settings::deserialize(j, "bodies", o.mBodies);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "terrainProjectionType", o.mTerrainProjectionType);
  cs::core::Settings::serialize(j, "lodFactor", o.mLODFactor);
  cs::core::Settings::serialize(j, "autoLod", o.mAutoLOD);
  cs::core::Settings::serialize(j, "textureGamma", o.mTextureGamma);
  cs::core::Settings::serialize(j, "enableHeightlines", o.mEnableHeightlines);
  cs::core::Settings::serialize(j, "enableLatLongGrid", o.mEnableLatLongGrid);
  cs::core::Settings::serialize(j, "colorMappingType", o.mColorMappingType);
  cs::core::Settings::serialize(j, "terrainColorMap", o.mTerrainColorMap);
  cs::core::Settings::serialize(j, "enableColorMixing", o.mEnableColorMixing);
  cs::core::Settings::serialize(j, "heightRange", o.mHeightRange);
  cs::core::Settings::serialize(j, "slopeRange", o.mSlopeRange);
  cs::core::Settings::serialize(j, "enableWireframe", o.mEnableWireframe);
  cs::core::Settings::serialize(j, "enableTilesDebug", o.mEnableTilesDebug);
  cs::core::Settings::serialize(j, "enableTilesFreeze", o.mEnableTilesFreeze);
  cs::core::Settings::serialize(j, "maxGPUTilesColor", o.mMaxGPUTilesColor);
  cs::core::Settings::serialize(j, "maxGPUTilesGray", o.mMaxGPUTilesGray);
  cs::core::Settings::serialize(j, "maxGPUTilesDEM", o.mMaxGPUTilesDEM);
  cs::core::Settings::serialize(j, "mapCache", o.mMapCache);
  cs::core::Settings::serialize(j, "bodies", o.mBodies);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });

  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-lod-bodies"] = *mPluginSettings; });

  mGuiManager->addPluginTabToSideBarFromHTML(
      "Body Settings", "landscape", "../share/resources/gui/lod_body_tab.html");
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Body Settings", "landscape", "../share/resources/gui/lod_body_settings.html");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-lod-bodies.js");

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

  mGuiManager->getGui()->registerCallback("lodBodies.setEnableColorMixing",
      "When enabled, the values of the colormap will be multiplied with the image channel.",
      std::function([this](bool enable) { mPluginSettings->mEnableColorMixing = enable; }));
  mPluginSettings->mEnableColorMixing.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("lodBodies.setEnableColorMixing", enable);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setTerrainLod",
      "Specifies the amount of detail of the planet's surface. Should be in the range 1-100.",
      std::function(
          [this](double value) { mPluginSettings->mLODFactor = static_cast<float>(value); }));
  mPluginSettings->mLODFactor.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("lodBodies.setTerrainLod", value); });

  mGuiManager->getGui()->registerCallback("lodBodies.setEnableAutoTerrainLod",
      "If set to true, the level-of-detail will be chosen automatically based on the current "
      "rendering performance.",
      std::function([this](bool enable) { mPluginSettings->mAutoLOD = enable; }));
  mPluginSettings->mAutoLOD.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("lodBodies.setEnableAutoTerrainLod", enable);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setTextureGamma",
      "A multiplier for the brightness of the image channel.", std::function([this](double value) {
        mPluginSettings->mTextureGamma = static_cast<float>(value);
      }));
  mPluginSettings->mTextureGamma.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("lodBodies.setTextureGamma", value); });

  mGuiManager->getGui()->registerCallback("lodBodies.setHeightRange",
      "Sets one end of the height range for the color mapping. The first parameter is the actual "
      "value, the second specifies which end to set: Zero for the lower end; One for the upper "
      "end.",
      std::function([this](double val, double handle) {
        auto range = mPluginSettings->mHeightRange.get();
        if (handle == 0.0) {
          range.x = static_cast<float>(val * 1000);
        } else {
          range.y = static_cast<float>(val * 1000);
        }
        mPluginSettings->mHeightRange = range;
      }));
  mPluginSettings->mHeightRange.connectAndTouch([this](glm::vec2 const& value) {
    mGuiManager->setSliderValue("lodBodies.setHeightRange", value);
  });

  mGuiManager->getGui()->registerCallback("lodBodies.setSlopeRange",
      "Sets one end of the slope range for the color mapping. The first parameter is the actual "
      "value, the second specifies which end to set: Zero for the lower end; One for the upper "
      "end.",
      std::function([this](double val, double handle) {
        auto range = mPluginSettings->mSlopeRange.get();
        if (handle == 0.0) {
          range.x = static_cast<float>(cs::utils::convert::toRadians(val));
        } else {
          range.y = static_cast<float>(cs::utils::convert::toRadians(val));
        }
        mPluginSettings->mSlopeRange = range;
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
        auto body = std::dynamic_pointer_cast<LodBody>(mSolarSystem->pActiveBody.get());
        if (body) {
          setImageSource(body, name);
        }
      }));

  mGuiManager->getGui()->registerCallback("lodBodies.setTilesDem",
      "Set the current planet's elevation channel to the TileSource with the given name.",
      std::function([this](std::string&& name) {
        auto body = std::dynamic_pointer_cast<LodBody>(mSolarSystem->pActiveBody.get());
        if (body) {
          setElevationSource(body, name);
        }
      }));

  mActiveBodyConnection = mSolarSystem->pActiveBody.connectAndTouch(
      [this](std::shared_ptr<cs::scene::CelestialBody> const& body) {
        auto lodBody = std::dynamic_pointer_cast<LodBody>(body);

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.sidebar.setTabEnabled", "Body Settings", lodBody != nullptr);

        if (!lodBody) {
          return;
        }

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.clearDropdown", "lodBodies.setTilesImg");
        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.clearDropdown", "lodBodies.setTilesDem");
        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "lodBodies.setTilesImg", "None", "None", "false");

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
      });

  mNonAutoLod = mPluginSettings->mLODFactor.get();

  mPluginSettings->mAutoLOD.connect([this](bool enabled) {
    if (enabled) {
      mNonAutoLod = mPluginSettings->mLODFactor.get();
    } else {
      mPluginSettings->mLODFactor = mNonAutoLod;
      mGuiManager->getGui()->callJavascript(
          "CosmoScout.gui.setSliderValue", "lodBodies.setTerrainLod", false, mNonAutoLod);
    }
  });

  mPluginSettings->mLODFactor.connect([this](float value) {
    if (mPluginSettings->mAutoLOD()) {
      mGuiManager->getGui()->callJavascript(
          "CosmoScout.gui.setSliderValue", "lodBodies.setTerrainLod", false, value);
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

  for (auto const& body : mLodBodies) {
    mInputManager->unregisterSelectable(body.second);
    mSolarSystem->unregisterBody(body.second);
  }

  mSolarSystem->pActiveBody.disconnect(mActiveBodyConnection);

  mGuiManager->removePluginTab("Body Settings");
  mGuiManager->removeSettingsSection("Body Settings");

  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableTilesFreeze");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableTilesDebug");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableWireframe");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableHeightlines");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableLatLongGrid");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableLatLongGridLabels");
  mGuiManager->getGui()->unregisterCallback("lodBodies.setEnableColorMixing");
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

    double minLODFactor = 15.0;
    double maxLODFactor = 50.0;
    double minTime      = 13.5;
    double maxTime      = 14.5;

    if (mFrameTimings->pFrameTime.get() > maxTime) {
      mPluginSettings->mLODFactor = static_cast<float>(std::max(
          minLODFactor, mPluginSettings->mLODFactor.get() -
                            std::min(1.0, 0.1 * (mFrameTimings->pFrameTime.get() - maxTime))));
    } else if (mFrameTimings->pFrameTime.get() < minTime) {
      mPluginSettings->mLODFactor = static_cast<float>(std::min(
          maxLODFactor, mPluginSettings->mLODFactor.get() +
                            std::min(1.0, 0.02 * (minTime - mFrameTimings->pFrameTime.get()))));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-lod-bodies"), *mPluginSettings);

  // For now, we cannot re-create the GLResources.
  if (!mGLResources) {
    mGLResources =
        std::make_shared<csp::lodbodies::GLResources>(mPluginSettings->mMaxGPUTilesDEM.get(),
            mPluginSettings->mMaxGPUTilesGray.get(), mPluginSettings->mMaxGPUTilesColor.get());

    mPluginSettings->mMaxGPUTilesColor.connect([](uint32_t /*val*/) {
      logger().warn("Changing the maximum number of allocated color tiles at run-time is not "
                    "supported. Please restart CosmoScout VR!");
    });

    mPluginSettings->mMaxGPUTilesGray.connect([](uint32_t /*val*/) {
      logger().warn("Changing the maximum number of allocated gray-scale tiles at run-time is not "
                    "supported. Please restart CosmoScout VR!");
    });

    mPluginSettings->mMaxGPUTilesDEM.connect([](uint32_t /*val*/) {
      logger().warn("Changing the maximum number of allocated elevation tiles at run-time is not "
                    "supported. Please restart CosmoScout VR!");
    });
  }

  // First try to re-configure existing lodBodies. We assume that they are similar if they have
  // the same name in the settings (which means they are attached to an anchor with the same name).
  auto lodBody = mLodBodies.begin();
  while (lodBody != mLodBodies.end()) {
    auto settings = mPluginSettings->mBodies.find(lodBody->first);
    if (settings != mPluginSettings->mBodies.end()) {
      // If there are settings for this lodBody, reconfigure it.
      auto anchor                           = mAllSettings->mAnchors.find(settings->first);
      auto [tStartExistence, tEndExistence] = anchor->second.getExistence();
      lodBody->second->setStartExistence(tStartExistence);
      lodBody->second->setEndExistence(tEndExistence);
      lodBody->second->setCenterName(anchor->second.mCenter);
      lodBody->second->setFrameName(anchor->second.mFrame);

      setImageSource(lodBody->second, settings->second.mActiveImgDataset);
      setElevationSource(lodBody->second, settings->second.mActiveDemDataset);

      ++lodBody;
    } else {
      // Else delete it.
      mSolarSystem->unregisterBody(lodBody->second);
      mInputManager->unregisterSelectable(lodBody->second);
      lodBody = mLodBodies.erase(lodBody);
    }
  }

  // Then add new lodBodies.
  for (auto const& settings : mPluginSettings->mBodies) {

    // Skip already existing bodies.
    if (mLodBodies.find(settings.first) != mLodBodies.end()) {
      continue;
    }

    auto anchor = mAllSettings->mAnchors.find(settings.first);

    if (anchor == mAllSettings->mAnchors.end()) {
      throw std::runtime_error(
          "There is no Anchor \"" + settings.first + "\" defined in the settings.");
    }

    auto [tStartExistence, tEndExistence] = anchor->second.getExistence();

    auto body = std::make_shared<LodBody>(mAllSettings, mGraphicsEngine, mSolarSystem,
        mPluginSettings, mGuiManager, anchor->second.mCenter, anchor->second.mFrame, mGLResources,
        tStartExistence, tEndExistence);

    mLodBodies.emplace(settings.first, body);

    setImageSource(body, settings.second.mActiveImgDataset);
    setElevationSource(body, settings.second.mActiveDemDataset);

    body->setSun(mSolarSystem->getSun());

    mSolarSystem->registerBody(body);
    mInputManager->registerSelectable(body);
  }

  mSolarSystem->pActiveBody.touch(mActiveBodyConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Plugin::Settings::Body& Plugin::getBodySettings(std::shared_ptr<LodBody> const& body) const {
  auto name = std::find_if(
      mLodBodies.begin(), mLodBodies.end(), [&](auto const& pair) { return pair.second == body; });
  return mPluginSettings->mBodies.at(name->first);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setImageSource(std::shared_ptr<LodBody> const& body, std::string const& name) const {
  auto& settings             = getBodySettings(body);
  settings.mActiveImgDataset = name;

  if (name == "None") {
    body->setIMGtileSource(nullptr);
    mGuiManager->getGui()->callJavascript("CosmoScout.lodBodies.setMapDataCopyright", "");
  } else {
    auto dataset = settings.mImgDatasets.find(name);
    if (dataset == settings.mImgDatasets.end()) {
      logger().warn("Cannot set image dataset '{}': There is no dataset defined with this name! "
                    "Using first dataset instead...",
          name);
      dataset = settings.mImgDatasets.begin();
    }

    auto source = std::make_shared<TileSourceWebMapService>();
    source->setCacheDirectory(mPluginSettings->mMapCache.get());
    source->setMaxLevel(dataset->second.mMaxLevel);
    source->setLayers(dataset->second.mLayers);
    source->setUrl(dataset->second.mURL);
    source->setDataType(dataset->second.mFormat);

    body->setIMGtileSource(source);

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

  settings.mActiveDemDataset = name;

  auto source = std::make_shared<TileSourceWebMapService>();
  source->setCacheDirectory(mPluginSettings->mMapCache.get());
  source->setMaxLevel(dataset->second.mMaxLevel);
  source->setLayers(dataset->second.mLayers);
  source->setUrl(dataset->second.mURL);
  source->setDataType(dataset->second.mFormat);

  body->setDEMtileSource(source);

  mGuiManager->getGui()->callJavascript(
      "CosmoScout.lodBodies.setElevationDataCopyright", dataset->second.mCopyright);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
