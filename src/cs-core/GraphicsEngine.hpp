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
  utils::Property<float> pApproximateSceneBrightness = 1.f;
  utils::Property<float> pAverageLuminance           = 1.f;
  utils::Property<float> pMaximumLuminance           = 1.f;

  GraphicsEngine(std::shared_ptr<Settings> const& settings);
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

  std::shared_ptr<core::Settings>               mSettings;
  std::shared_ptr<graphics::ShadowMap>          mShadowMap;
  std::shared_ptr<graphics::HDRBuffer>          mHDRBuffer;
  std::shared_ptr<graphics::ClearHDRBufferNode> mClearNode;
  std::shared_ptr<graphics::ToneMappingNode>    mToneMappingNode;
};

} // namespace cs::core

#endif // CS_CORE_GRAPHICS_GraphicsEngine_HPP
