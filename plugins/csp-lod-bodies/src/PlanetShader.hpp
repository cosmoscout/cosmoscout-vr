////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_PLANET_SHADER_HPP
#define CSP_LOD_BODIES_PLANET_SHADER_HPP

#include "../../../src/cs-core/EclipseShadowReceiver.hpp"
#include "../../../src/cs-graphics/ColorMap.hpp"
#include "../../../src/cs-utils/Property.hpp"

#include "Plugin.hpp"
#include "TerrainShader.hpp"

#include <glm/glm.hpp>
#include <vector>

class VistaTexture;

namespace cs::core {
class GuiManager;
class Settings;
} // namespace cs::core

namespace csp::lodbodies {

/// The shader for rendering a planet.
class PlanetShader : public TerrainShader {
 public:
  cs::utils::Property<bool> pEnableTexture = true; ///< If false the image data will not be drawn.

  PlanetShader(std::shared_ptr<cs::core::Settings>     settings,
      std::shared_ptr<Plugin::Settings>                pluginSettings,
      std::shared_ptr<cs::core::GuiManager>            pGuiManager,
      std::shared_ptr<cs::core::EclipseShadowReceiver> eclipseShadowReceiver);

  PlanetShader(PlanetShader const& other) = delete;
  PlanetShader(PlanetShader&& other)      = delete;

  PlanetShader& operator=(PlanetShader const& other) = delete;
  PlanetShader& operator=(PlanetShader&& other) = delete;

  ~PlanetShader() override;

  /// This is required to get several properties from the plugin settings.
  void               setObjectName(std::string objectName);
  std::string const& getObjectName() const;

  void setSun(glm::vec3 direction, float illuminance);

  void bind() override;
  void release() override;

 private:
  void compile() override;

  std::shared_ptr<cs::core::Settings>              mSettings;
  std::shared_ptr<cs::core::GuiManager>            mGuiManager;
  std::shared_ptr<Plugin::Settings>                mPluginSettings;
  std::shared_ptr<cs::core::EclipseShadowReceiver> mEclipseShadowReceiver;
  std::string                                      mObjectName;
  glm::vec3                                        mSunDirection             = glm::vec3(0, 1, 0);
  float                                            mSunIlluminance           = 1.F;
  VistaTexture*                                    mFontTexture              = nullptr;
  int                                              mEnableLightingConnection = -1;
  int                                              mEnableShadowsDebugConnection = -1;
  int                                              mEnableShadowsConnection      = -1;
  int                                              mLightingQualityConnection    = -1;
  int                                              mEnableHDRConnection          = -1;

  static std::map<std::string, cs::graphics::ColorMap> mColorMaps;
};

} // namespace csp::lodbodies

#endif // CS_CORE_PLANET_SHADER_HPP
