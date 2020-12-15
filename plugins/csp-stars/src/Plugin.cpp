////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::stars::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::stars {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "celestialGridTexture", o.mCelestialGridTexture);
  cs::core::Settings::deserialize(j, "starFiguresTexture", o.mStarFiguresTexture);
  cs::core::Settings::deserialize(j, "celestialGridColor", o.mCelestialGridColor);
  cs::core::Settings::deserialize(j, "starFiguresColor", o.mStarFiguresColor);
  cs::core::Settings::deserialize(j, "starTexture", o.mStarTexture);
  cs::core::Settings::deserialize(j, "cacheFile", o.mCacheFile);
  cs::core::Settings::deserialize(j, "hipparcosCatalog", o.mHipparcosCatalog);
  cs::core::Settings::deserialize(j, "tychoCatalog", o.mTychoCatalog);
  cs::core::Settings::deserialize(j, "tycho2Catalog", o.mTycho2Catalog);
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "enableCelestialGrid", o.mEnableCelestialGrid);
  cs::core::Settings::deserialize(j, "enableStarFigures", o.mEnableStarFigures);
  cs::core::Settings::deserialize(j, "luminanceMultiplicator", o.mLuminanceMultiplicator);
  cs::core::Settings::deserialize(j, "drawMode", o.mDrawMode);
  cs::core::Settings::deserialize(j, "size", o.mSize);
  cs::core::Settings::deserialize(j, "magnitudeRange", o.mMagnitudeRange);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "celestialGridTexture", o.mCelestialGridTexture);
  cs::core::Settings::serialize(j, "starFiguresTexture", o.mStarFiguresTexture);
  cs::core::Settings::serialize(j, "celestialGridColor", o.mCelestialGridColor);
  cs::core::Settings::serialize(j, "starFiguresColor", o.mStarFiguresColor);
  cs::core::Settings::serialize(j, "starTexture", o.mStarTexture);
  cs::core::Settings::serialize(j, "cacheFile", o.mCacheFile);
  cs::core::Settings::serialize(j, "hipparcosCatalog", o.mHipparcosCatalog);
  cs::core::Settings::serialize(j, "tychoCatalog", o.mTychoCatalog);
  cs::core::Settings::serialize(j, "tycho2Catalog", o.mTycho2Catalog);
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "enableCelestialGrid", o.mEnableCelestialGrid);
  cs::core::Settings::serialize(j, "enableStarFigures", o.mEnableStarFigures);
  cs::core::Settings::serialize(j, "luminanceMultiplicator", o.mLuminanceMultiplicator);
  cs::core::Settings::serialize(j, "drawMode", o.mDrawMode);
  cs::core::Settings::serialize(j, "size", o.mSize);
  cs::core::Settings::serialize(j, "magnitudeRange", o.mMagnitudeRange);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-stars"] = mPluginSettings; });

  // Create the Stars object based on the settings.
  mStars = std::make_unique<Stars>();

  // Add the stars to the scenegraph.
  mStarsTransform = std::make_shared<cs::scene::CelestialAnchorNode>(
      mSceneGraph->GetRoot(), mSceneGraph->GetNodeBridge(), "", "Solar System Barycenter", "J2000");
  mSolarSystem->registerAnchor(mStarsTransform);

  mSceneGraph->GetRoot()->AddChild(mStarsTransform.get());

  mStarsNode.reset(mSceneGraph->NewOpenGLNode(mStarsTransform.get(), mStars.get()));

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mStarsTransform.get(), static_cast<int>(cs::utils::DrawOrder::eStars));

  // Configure the stars node when a public property is changed.
  mPluginSettings.mEnabled.connect([this](bool val) { mStarsNode->SetIsEnabled(val); });
  mPluginSettings.mDrawMode.connect([this](Stars::DrawMode val) { mStars->setDrawMode(val); });
  mPluginSettings.mSize.connect([this](float val) { mStars->setSolidAngle(val * 0.0001F); });
  mPluginSettings.mMagnitudeRange.connect([this](glm::vec2 const& val) {
    mStars->setMinMagnitude(val.x);
    mStars->setMaxMagnitude(val.y);
  });

  // Add the stars user interface components to the CosmoScout user interface.
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Stars", "star", "../share/resources/gui/stars_settings.html");

  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-stars.js");

  // Register JavaScript callbacks.
  mGuiManager->getGui()->registerCallback("stars.setEnabled",
      "Enables or disables the rendering of stars.",
      std::function([this](bool enable) { mPluginSettings.mEnabled = enable; }));
  mPluginSettings.mEnabled.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("stars.setEnabled", enable); });

  mGuiManager->getGui()->registerCallback("stars.setEnableGrid",
      "If stars are enabled, this enables the rendering of a background grid in celestial "
      "coordinates.",
      std::function([this](bool enable) { mPluginSettings.mEnableCelestialGrid = enable; }));
  mPluginSettings.mEnableCelestialGrid.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("stars.setEnableGrid", enable); });

  mGuiManager->getGui()->registerCallback("stars.setEnableFigures",
      "If stars are enabled, this enables the rendering of star figures.",
      std::function([this](bool enable) { mPluginSettings.mEnableStarFigures = enable; }));
  mPluginSettings.mEnableStarFigures.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("stars.setEnableFigures", enable); });

  mGuiManager->getGui()->registerCallback("stars.setLuminanceBoost",
      "Adds an artificial brightness boost to the stars.", std::function([this](double value) {
        mPluginSettings.mLuminanceMultiplicator = static_cast<float>(value);
      }));
  mPluginSettings.mLuminanceMultiplicator.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("stars.setLuminanceBoost", value); });

  mGuiManager->getGui()->registerCallback("stars.setSize",
      "Sets the apparent size of stars on screen.",
      std::function([this](double value) { mPluginSettings.mSize = static_cast<float>(value); }));
  mPluginSettings.mSize.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("stars.setSize", value); });

  mGuiManager->getGui()->registerCallback("stars.setMagnitude",
      "Sets the magnitude range for stars.", std::function([this](double val1, double val2) {
        mPluginSettings.mMagnitudeRange = glm::vec2(val1, val2);
      }));
  mPluginSettings.mMagnitudeRange.connectAndTouch(
      [this](glm::vec2 const& value) { mGuiManager->setSliderValue("stars.setMagnitude", value); });

  mGuiManager->getGui()->registerCallback("stars.setDrawMode0",
      "Enables point draw mode for the stars.",
      std::function([this]() { mPluginSettings.mDrawMode = Stars::DrawMode::ePoint; }));
  mGuiManager->getGui()->registerCallback("stars.setDrawMode1",
      "Enables smooth point draw mode for the stars.",
      std::function([this]() { mPluginSettings.mDrawMode = Stars::DrawMode::eSmoothPoint; }));
  mGuiManager->getGui()->registerCallback("stars.setDrawMode2",
      "Enables disc draw mode for the stars.",
      std::function([this]() { mPluginSettings.mDrawMode = Stars::DrawMode::eDisc; }));
  mGuiManager->getGui()->registerCallback("stars.setDrawMode3",
      "Enables smooth disc draw mode for the stars.",
      std::function([this]() { mPluginSettings.mDrawMode = Stars::DrawMode::eSmoothDisc; }));
  mGuiManager->getGui()->registerCallback("stars.setDrawMode4",
      "Enables scaled disc draw mode for the stars.",
      std::function([this]() { mPluginSettings.mDrawMode = Stars::DrawMode::eScaledDisc; }));
  mGuiManager->getGui()->registerCallback("stars.setDrawMode5",
      "Enables sprite draw mode for the stars.",
      std::function([this]() { mPluginSettings.mDrawMode = Stars::DrawMode::eSprite; }));
  mPluginSettings.mDrawMode.connect([this](Stars::DrawMode drawMode) {
    if (drawMode == Stars::DrawMode::ePoint) {
      mGuiManager->setRadioChecked("stars.setDrawMode0");
    } else if (drawMode == Stars::DrawMode::eSmoothPoint) {
      mGuiManager->setRadioChecked("stars.setDrawMode1");
    } else if (drawMode == Stars::DrawMode::eDisc) {
      mGuiManager->setRadioChecked("stars.setDrawMode2");
    } else if (drawMode == Stars::DrawMode::eSmoothDisc) {
      mGuiManager->setRadioChecked("stars.setDrawMode3");
    } else if (drawMode == Stars::DrawMode::eScaledDisc) {
      mGuiManager->setRadioChecked("stars.setDrawMode4");
    } else if (drawMode == Stars::DrawMode::eSprite) {
      mGuiManager->setRadioChecked("stars.setDrawMode5");
    }
  });

  mEnableHDRConnection = mAllSettings->mGraphics.pEnableHDR.connectAndTouch(
      [this](bool value) { mStars->setEnableHDR(value); });

  // Load settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mSolarSystem->unregisterAnchor(mStarsTransform);
  mSceneGraph->GetRoot()->DisconnectChild(mStarsTransform.get());

  mAllSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);

  mGuiManager->removeSettingsSection("Stars");

  mGuiManager->getGui()->callJavascript("CosmoScout.removeApi", "stars");

  mGuiManager->getGui()->unregisterCallback("stars.setLuminanceBoost");
  mGuiManager->getGui()->unregisterCallback("stars.setSize");
  mGuiManager->getGui()->unregisterCallback("stars.setMagnitude");
  mGuiManager->getGui()->unregisterCallback("stars.setDrawMode0");
  mGuiManager->getGui()->unregisterCallback("stars.setDrawMode1");
  mGuiManager->getGui()->unregisterCallback("stars.setDrawMode2");
  mGuiManager->getGui()->unregisterCallback("stars.setDrawMode3");
  mGuiManager->getGui()->unregisterCallback("stars.setDrawMode4");
  mGuiManager->getGui()->unregisterCallback("stars.setEnabled");
  mGuiManager->getGui()->unregisterCallback("stars.setEnableGrid");
  mGuiManager->getGui()->unregisterCallback("stars.setEnableFigures");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {

  // Update the stars brightness based on the scene's pApproximateSceneBrightness. This is to fade
  // out the stars when we are close to a Planet. If HDR rendering is enabled, we will not change
  // the star's brightness.
  float fIntensity = mGraphicsEngine->pApproximateSceneBrightness.get();

  if (mAllSettings->mGraphics.pEnableHDR.get()) {
    fIntensity = 1.F;
  }

  mStars->setLuminanceMultiplicator(
      fIntensity * std::exp(mPluginSettings.mLuminanceMultiplicator.get()));
  mStars->setCelestialGridColor(VistaColor(0.5F, 0.8F, 1.F,
      0.3F * fIntensity * (mPluginSettings.mEnableCelestialGrid.get() ? 1.F : 0.F)));
  mStars->setStarFiguresColor(VistaColor(
      0.5F, 1.F, 0.8F, 0.3F * fIntensity * (mPluginSettings.mEnableStarFigures.get() ? 1.F : 0.F)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-stars"), mPluginSettings);

  // Configure the stars based on the settings.
  mStars->setStarTexture(mPluginSettings.mStarTexture);
  mStars->setCelestialGridTexture(mPluginSettings.mCelestialGridTexture.get());
  mStars->setStarFiguresTexture(mPluginSettings.mStarFiguresTexture.get());

  auto const& bg1 = mPluginSettings.mCelestialGridColor.get();
  auto const& bg2 = mPluginSettings.mStarFiguresColor.get();
  mStars->setCelestialGridColor(VistaColor(bg1.r, bg1.g, bg1.b, bg1.a));
  mStars->setStarFiguresColor(VistaColor(bg2.r, bg2.g, bg2.b, bg2.a));

  mStars->setCacheFile(mPluginSettings.mCacheFile.value_or("star_cache.dat"));

  std::map<Stars::CatalogType, std::string> catalogs;

  if (mPluginSettings.mHipparcosCatalog) {
    catalogs[Stars::CatalogType::eHipparcos] = *mPluginSettings.mHipparcosCatalog;
  }

  if (mPluginSettings.mTychoCatalog) {
    catalogs[Stars::CatalogType::eTycho] = *mPluginSettings.mTychoCatalog;
  }

  if (mPluginSettings.mTycho2Catalog) {
    catalogs[Stars::CatalogType::eTycho2] = *mPluginSettings.mTycho2Catalog;
  }

  mStars->setCatalogs(catalogs);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::stars
