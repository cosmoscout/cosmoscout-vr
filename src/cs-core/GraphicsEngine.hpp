////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_GRAPHICS_GraphicsEngine_HPP
#define CS_CORE_GRAPHICS_GraphicsEngine_HPP

#include "../cs-graphics/HDRBuffer.hpp"
#include "../cs-graphics/Shadows.hpp"
#include "../cs-utils/Property.hpp"
#include "Settings.hpp"

#include <glm/glm.hpp>
#include <memory>

namespace cs::graphics {
class ClearHDRBufferNode;
class ToneMappingNode;
} // namespace cs::graphics

namespace cs::core {

/// The GraphicsEngine is responsible for managing the ShadowMap. It also provides access to global
/// render settings. This class should only be instantiated once - this instance will be passed to
/// all plugins.
class CS_CORE_EXPORT GraphicsEngine {
 public:
  utils::Property<float>     pHeightScale                = 1.f;
  utils::Property<float>     pWidgetScale                = 1.f;
  utils::Property<bool>      pEnableLighting             = false;
  utils::Property<bool>      pEnableHDR                  = false;
  utils::Property<int>       pLightingQuality            = 2;
  utils::Property<bool>      pEnableShadows              = false;
  utils::Property<bool>      pEnableShadowsDebug         = false;
  utils::Property<bool>      pEnableShadowsFreeze        = false;
  utils::Property<int>       pShadowMapResolution        = 2048;
  utils::Property<int>       pShadowMapCascades          = 3;
  utils::Property<float>     pShadowMapBias              = 1.0f;
  utils::Property<glm::vec2> pShadowMapRange             = glm::vec2(0.f, 100.f);
  utils::Property<glm::vec2> pShadowMapExtension         = glm::vec2(-100.f, 100.f);
  utils::Property<float>     pShadowMapSplitDistribution = 1.f;
  utils::Property<bool>      pEnableAutoExposure         = true;
  utils::Property<float>     pExposure                   = 0.f;                    // in EV
  utils::Property<glm::vec2> pAutoExposureRange          = glm::vec2(-14.f, 10.f); // in EV
  utils::Property<float>     pExposureCompensation       = 0.f;                    // in EV
  utils::Property<float>     pExposureAdaptionSpeed      = 3.f;
  utils::Property<float>     pSensorDiagonal             = 42.f; // in millimeters
  utils::Property<float>     pFocalLength                = 24.f; // in millimeters
  utils::Property<float>     pAmbientBrightness          = std::pow(0.25f, 10.f);
  utils::Property<bool>      pEnableAutoGlow             = true;
  utils::Property<float>     pGlowIntensity              = 0.5f;
  utils::Property<graphics::ExposureMeteringMode> pExposureMeteringMode =
      graphics::ExposureMeteringMode::AVERAGE;

  utils::Property<float> pApproximateSceneBrightness = 1.f;
  utils::Property<float> pAverageLuminance           = 1.f;
  utils::Property<float> pMaximumLuminance           = 1.f;

  GraphicsEngine(std::shared_ptr<const Settings> const& settings);
  ~GraphicsEngine();

  /// All objects which are able to cast shadows need to be registered.
  void registerCaster(graphics::ShadowCaster* caster);
  void unregisterCaster(graphics::ShadowCaster* caster);

  /// The light direction in world space.
  void update(glm::vec3 const& sunDirection);

  std::shared_ptr<graphics::ShadowMap> getShadowMap() const;
  std::shared_ptr<graphics::HDRBuffer> getHDRBuffer() const;

  void enableGLDebug(bool onlyErrors = true);
  void disableGLDebug();

 private:
  void calculateCascades();

  std::shared_ptr<const core::Settings>         mSettings;
  std::shared_ptr<graphics::ShadowMap>          mShadowMap;
  std::shared_ptr<graphics::HDRBuffer>          mHDRBuffer;
  std::shared_ptr<graphics::ClearHDRBufferNode> mClearNode;
  std::shared_ptr<graphics::ToneMappingNode>    mToneMappingNode;
};

} // namespace cs::core

#endif // CS_CORE_GRAPHICS_GraphicsEngine_HPP
