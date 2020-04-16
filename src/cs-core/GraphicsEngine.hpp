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
  utils::Property<float>     pHeightScale                = 1.F;
  utils::Property<float>     pWidgetScale                = 1.F;
  utils::Property<bool>      pEnableLighting             = false;
  utils::Property<bool>      pEnableHDR                  = false;
  utils::Property<int>       pLightingQuality            = 2;
  utils::Property<bool>      pEnableShadows              = false;
  utils::Property<bool>      pEnableShadowsDebug         = false;
  utils::Property<bool>      pEnableShadowsFreeze        = false;
  utils::Property<int>       pShadowMapResolution        = 2048;
  utils::Property<int>       pShadowMapCascades          = 3;
  utils::Property<float>     pShadowMapBias              = 1.0F;
  utils::Property<glm::vec2> pShadowMapRange             = glm::vec2(0.F, 100.F);
  utils::Property<glm::vec2> pShadowMapExtension         = glm::vec2(-100.F, 100.F);
  utils::Property<float>     pShadowMapSplitDistribution = 1.F;
  utils::Property<bool>      pEnableAutoExposure         = true;
  utils::Property<float>     pExposure                   = 0.F;                    // in EV
  utils::Property<glm::vec2> pAutoExposureRange          = glm::vec2(-14.F, 10.F); // in EV
  utils::Property<float>     pExposureCompensation       = 0.F;                    // in EV
  utils::Property<float>     pExposureAdaptionSpeed      = 3.F;
  utils::Property<float>     pSensorDiagonal             = 42.F; // in millimeters
  utils::Property<float>     pFocalLength                = 24.F; // in millimeters
  utils::Property<float>     pAmbientBrightness          = std::pow(0.25F, 10.F);
  utils::Property<bool>      pEnableAutoGlow             = true;
  utils::Property<float>     pGlowIntensity              = 0.5F;
  utils::Property<float>     pApproximateSceneBrightness = 1.F;
  utils::Property<float>     pAverageLuminance           = 1.F;
  utils::Property<float>     pMaximumLuminance           = 1.F;

  explicit GraphicsEngine(std::shared_ptr<const Settings> const& settings);

  GraphicsEngine(GraphicsEngine const& other) = delete;
  GraphicsEngine(GraphicsEngine&& other)      = delete;

  GraphicsEngine& operator=(GraphicsEngine const& other) = delete;
  GraphicsEngine& operator=(GraphicsEngine&& other) = delete;

  ~GraphicsEngine();

  /// All objects which are able to cast shadows need to be registered.
  void registerCaster(graphics::ShadowCaster* caster);
  void unregisterCaster(graphics::ShadowCaster* caster);

  /// The light direction in world space.
  void update(glm::vec3 const& sunDirection);

  std::shared_ptr<graphics::ShadowMap> getShadowMap() const;
  std::shared_ptr<graphics::HDRBuffer> getHDRBuffer() const;

  static void enableGLDebug(bool onlyErrors = true);
  static void disableGLDebug();

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
