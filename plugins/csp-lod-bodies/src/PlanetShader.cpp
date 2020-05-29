////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "PlanetShader.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-gui/GuiItem.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaOGLUtils.h>
#include <VistaOGLExt/VistaShaderRegistry.h>

#include <utility>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
GLenum const TEXUNITNAMEFONT = GL_TEXTURE10;
GLenum const TEXUNITNAMELUT  = GL_TEXTURE11;
GLint const  TEXUNITFONT     = 10;
GLint const  TEXUNITLUT      = 11;
} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

std::map<std::string, cs::graphics::ColorMap> PlanetShader::mColorMaps;

////////////////////////////////////////////////////////////////////////////////////////////////////

PlanetShader::PlanetShader(std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<Plugin::Settings>                          pluginSettings,
    std::shared_ptr<cs::core::GuiManager> const&               pGuiManager)
    : mSettings(std::move(settings))
    , mGuiManager(pGuiManager)
    , mPluginSettings(std::move(pluginSettings))
    , mFontTexture(VistaOGLUtils::LoadTextureFromTga("../share/resources/textures/font.tga")) {
  // clang-format off
    pTextureIsRGB.connect(
        [this](bool /*ignored*/) { mShaderDirty = true; });
    pEnableTexture.connect(
        [this](bool /*ignored*/) { mShaderDirty = true; });
    mPluginSettings->mEnableHeightlines.connect(
        [this](bool /*ignored*/) { mShaderDirty = true; });
    mPluginSettings->mColorMappingType.connect(
        [this](Plugin::Settings::ColorMappingType /*ignored*/) { mShaderDirty = true; });
    mPluginSettings->mTerrainProjectionType.connect(
        [this](Plugin::Settings::TerrainProjectionType /*ignored*/) { mShaderDirty = true; });
    mEnableLightingConnection = mSettings->mGraphics.pEnableLighting.connect(
        [this](bool /*ignored*/) { mShaderDirty = true; });
    mEnableShadowsDebugConnection = mSettings->mGraphics.pEnableShadowsDebug.connect(
        [this](bool /*ignored*/) { mShaderDirty = true; });
    mEnableShadowsConnection = mSettings->mGraphics.pEnableShadows.connect(
        [this](bool /*ignored*/) { mShaderDirty = true; });
    mEnableHDRConnection = mSettings->mGraphics.pEnableHDR.connect(
        [this](bool /*ignored*/) { mShaderDirty = true; });
    mLightingQualityConnection = mSettings->mGraphics.pLightingQuality.connect(
        [this](int /*ignored*/) { mShaderDirty = true; });
    mPluginSettings->mEnableTilesDebug.connect(
        [this](bool /*ignored*/) { mShaderDirty = true; });
    mPluginSettings->mEnableLatLongGrid.connect(
        [this](bool /*ignored*/) { mShaderDirty = true; });
    mPluginSettings->mEnableColorMixing.connect(
        [this](bool /*ignored*/) { mShaderDirty = true; });
  // clang-format on

  // TODO: color map mangement could be done in a separate class
  if (mColorMaps.empty()) {
    auto files(cs::utils::filesystem::listFiles("../share/resources/colormaps"));

    bool first = true;
    for (auto const& file : files) {
      std::string  name(file);
      const size_t lastSlashIdx = name.find_last_of("\\/");
      if (std::string::npos != lastSlashIdx) {
        name.erase(0, lastSlashIdx + 1);
      }

      mColorMaps.insert(std::make_pair(name, cs::graphics::ColorMap(file)));
      pGuiManager->getGui()->callJavascript(
          "CosmoScout.gui.addDropdownValue", "lodBodies.setColormap", name, name, first);
      if (first) {
        first                             = false;
        mPluginSettings->mTerrainColorMap = name;
      }
    }

    pGuiManager->getGui()->registerCallback("lodBodies.setColormap",
        "Make the planet shader use the colormap with the given name.",
        std::function([this](std::string&& name) { mPluginSettings->mTerrainColorMap = name; }));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

PlanetShader::~PlanetShader() {
  delete mFontTexture;
  mSettings->mGraphics.pEnableLighting.disconnect(mEnableLightingConnection);
  mSettings->mGraphics.pEnableShadowsDebug.disconnect(mEnableShadowsDebugConnection);
  mSettings->mGraphics.pEnableShadows.disconnect(mEnableShadowsConnection);
  mSettings->mGraphics.pLightingQuality.disconnect(mLightingQualityConnection);
  mSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);

  mGuiManager->getGui()->unregisterCallback("lodBodies.setColormap");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PlanetShader::setSun(glm::vec3 const& direction, float illuminance) {
  mSunDirection   = direction;
  mSunIlluminance = illuminance;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PlanetShader::compile() {
  VistaShaderRegistry& reg = VistaShaderRegistry::GetInstance();
  mVertexSource            = reg.RetrieveShader("Planet.vert");
  mFragmentSource          = reg.RetrieveShader("Planet.frag");

  cs::utils::replaceString(
      mFragmentSource, "$TEXTURE_IS_RGB", cs::utils::toString(pTextureIsRGB.get()));
  cs::utils::replaceString(mFragmentSource, "$SHOW_HEIGHT_LINES",
      cs::utils::toString(mPluginSettings->mEnableHeightlines.get()));
  cs::utils::replaceString(
      mFragmentSource, "$SHOW_TEXTURE", cs::utils::toString(pEnableTexture.get()));
  cs::utils::replaceString(mFragmentSource, "$COLOR_MAPPING_TYPE",
      cs::utils::toString(static_cast<int>(mPluginSettings->mColorMappingType.get())));
  cs::utils::replaceString(mFragmentSource, "$ENABLE_LIGHTING",
      cs::utils::toString(mSettings->mGraphics.pEnableLighting.get()));
  cs::utils::replaceString(
      mFragmentSource, "$ENABLE_HDR", cs::utils::toString(mSettings->mGraphics.pEnableHDR.get()));
  cs::utils::replaceString(mFragmentSource, "$ENABLE_SHADOWS_DEBUG",
      cs::utils::toString(mSettings->mGraphics.pEnableShadowsDebug.get()));
  cs::utils::replaceString(mFragmentSource, "$ENABLE_SHADOWS",
      cs::utils::toString(mSettings->mGraphics.pEnableShadows.get()));
  cs::utils::replaceString(mFragmentSource, "$LIGHTING_QUALITY",
      cs::utils::toString(mSettings->mGraphics.pLightingQuality.get()));
  cs::utils::replaceString(mFragmentSource, "$SHOW_TILE_BORDER",
      cs::utils::toString(mPluginSettings->mEnableTilesDebug.get()));
  cs::utils::replaceString(mFragmentSource, "$SHOW_LAT_LONG_LABELS",
      cs::utils::toString(mPluginSettings->mEnableLatLongGrid.get()));
  cs::utils::replaceString(mFragmentSource, "$SHOW_LAT_LONG",
      cs::utils::toString(mPluginSettings->mEnableLatLongGrid.get()));
  cs::utils::replaceString(mFragmentSource, "$MIX_COLORS",
      cs::utils::toString(mPluginSettings->mEnableColorMixing.get()));

  cs::utils::replaceString(mVertexSource, "$LIGHTING_QUALITY",
      cs::utils::toString(mSettings->mGraphics.pLightingQuality.get()));
  cs::utils::replaceString(mVertexSource, "$TERRAIN_PROJECTION_TYPE",
      cs::utils::toString(static_cast<int>(mPluginSettings->mTerrainProjectionType.get())));

  TerrainShader::compile();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PlanetShader::bind() {
  TerrainShader::bind();

  GLint loc = -1;
  loc       = mShader.GetUniformLocation("heightTex");
  mShader.SetUniform(loc, TEXUNITLUT);

  loc = mShader.GetUniformLocation("fontTex");
  mShader.SetUniform(loc, TEXUNITFONT);

  loc = mShader.GetUniformLocation("heightMin");
  mShader.SetUniform(loc, mPluginSettings->mHeightRange.get().x);

  loc = mShader.GetUniformLocation("heightMax");
  mShader.SetUniform(loc, mPluginSettings->mHeightRange.get().y);

  loc = mShader.GetUniformLocation("slopeMin");
  mShader.SetUniform(loc, mPluginSettings->mSlopeRange.get().x);

  loc = mShader.GetUniformLocation("slopeMax");
  mShader.SetUniform(loc, mPluginSettings->mSlopeRange.get().y);

  loc = mShader.GetUniformLocation("ambientBrightness");
  mShader.SetUniform(loc, mSettings->mGraphics.pAmbientBrightness.get());

  loc = mShader.GetUniformLocation("texGamma");
  mShader.SetUniform(loc, mPluginSettings->mTextureGamma.get());

  loc = mShader.GetUniformLocation("uSunDirIlluminance");
  mShader.SetUniform(loc, mSunDirection.x, mSunDirection.y, mSunDirection.z, mSunIlluminance);

  loc = mShader.GetUniformLocation("farClip");
  mShader.SetUniform(loc, cs::utils::getCurrentFarClipDistance());

  mFontTexture->Bind(TEXUNITNAMEFONT);

  auto it(mColorMaps.find(mPluginSettings->mTerrainColorMap.get()));
  if (it != mColorMaps.end()) {
    it->second.bind(TEXUNITNAMELUT);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PlanetShader::release() {
  auto it(mColorMaps.find(mPluginSettings->mTerrainColorMap.get()));
  if (it != mColorMaps.end()) {
    it->second.unbind(TEXUNITNAMELUT);
  }

  mFontTexture->Unbind(TEXUNITNAMEFONT);

  TerrainShader::release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
