////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_GRAPHICS_ENGINE_HPP
#define CS_CORE_GRAPHICS_ENGINE_HPP

#include "../cs-graphics/HDRBuffer.hpp"
#include "../cs-graphics/Shadows.hpp"
#include "../cs-utils/Property.hpp"
#include "Settings.hpp"

#include <glm/glm.hpp>
#include <memory>

namespace cs::graphics {
struct EclipseShadowMap;
class ClearHDRBufferNode;
class ToneMappingNode;
class SetupGLNode;
} // namespace cs::graphics

namespace cs::core {

/// The GraphicsEngine is responsible for managing the ShadowMap. It also provides access to global
/// render settings. This class should only be instantiated once - this instance will be passed to
/// all plugins.
class CS_CORE_EXPORT GraphicsEngine {

 public:
  utils::Property<float> pApproximateSceneBrightness = 0.F;
  utils::Property<float> pAverageLuminance           = 1.F;
  utils::Property<float> pMaximumLuminance           = 1.F;

  explicit GraphicsEngine(std::shared_ptr<Settings> settings);

  GraphicsEngine(GraphicsEngine const& other) = default;
  GraphicsEngine(GraphicsEngine&& other)      = default;

  GraphicsEngine& operator=(GraphicsEngine const& other) = default;
  GraphicsEngine& operator=(GraphicsEngine&& other)      = default;

  ~GraphicsEngine();

  /// All objects which are able to cast shadows need to be registered.
  void registerCaster(graphics::ShadowCaster* caster);
  void unregisterCaster(graphics::ShadowCaster* caster);

  /// The light direction in world space.
  void update(glm::vec3 const& sunDirection);

  std::shared_ptr<graphics::ShadowMap> getShadowMap() const;
  std::shared_ptr<graphics::HDRBuffer> getHDRBuffer() const;

  /// Returns a list of all available eclipse shadow maps. You can use the eclipse shadow API of the
  /// SolarSystem to get all relevant eclipse shadow maps for a given position in space.
  std::vector<std::shared_ptr<graphics::EclipseShadowMap>> const& getEclipseShadowMaps() const;

  /// Returns the current depth or color buffer as a texture which can be used as shader input.
  /// If HDR rendering is enabled, this simply returns the current read-targets. If HDR rendering is
  /// disabled, the current frame buffer contents are copied to a texture first. This copy is only
  /// done once per frame, subsequent calls will use the cached texture. If you set forceCopy to
  /// true, the texture will be copied again.
  VistaTexture* getCurrentDepthBufferAsTexture(bool forceCopy);
  VistaTexture* getCurrentColorBufferAsTexture(bool forceCopy);

 private:
  void calculateCascades();

  std::shared_ptr<core::Settings>                          mSettings;
  std::shared_ptr<graphics::ShadowMap>                     mShadowMap;
  std::shared_ptr<graphics::HDRBuffer>                     mHDRBuffer;
  std::shared_ptr<graphics::ClearHDRBufferNode>            mClearNode;
  std::shared_ptr<graphics::SetupGLNode>                   mSetupGLNode;
  std::shared_ptr<graphics::ToneMappingNode>               mToneMappingNode;
  std::vector<std::shared_ptr<graphics::EclipseShadowMap>> mEclipseShadowMaps;
  std::shared_ptr<VistaTexture>                            mFallbackEclipseShadowMap;

  struct ViewportData {
    // MSVC does not like a unique_ptr here. Let's use a shared_ptr instead, although it is not
    // really shared.
    std::shared_ptr<VistaTexture> mBuffer;
    bool                          mDirty = true;
  };

  // These are used to cache the frame buffer contents for the current viewport for the
  // getCurrentDepthBufferAsTexture and getCurrentColorBufferAsTexture calls. They are only
  // filled if the methods are called and HDR rendering is disabled.
  std::unordered_map<VistaViewport*, ViewportData> mDepthBuffers;
  std::unordered_map<VistaViewport*, ViewportData> mColorBuffers;
};

} // namespace cs::core

#endif // CS_CORE_GRAPHICS_ENGINE_HPP
