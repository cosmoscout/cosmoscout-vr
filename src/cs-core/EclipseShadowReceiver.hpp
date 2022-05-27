////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_ECLIPSE_SHADOW_RECEIVER_HPP
#define CS_CORE_ECLIPSE_SHADOW_RECEIVER_HPP

#include "cs_core_export.hpp"

#include <glm/glm.hpp>
#include <memory>
#include <vector>

class VistaGLSLShader;

namespace cs::scene {
class CelestialObject;
class CelestialObserver;
class CelestialBody;
} // namespace cs::scene

namespace cs::graphics {
struct EclipseShadowMap;
}

namespace cs::core {

class SolarSystem;
class Settings;

enum class EclipseShadowMode {
  eNone               = 0,
  eDebug              = 1,
  eLinear             = 2,
  eSmoothstep         = 3,
  eCircleIntersection = 4,
  eTexture            = 5,
  eFastTexture        = 6
};

class CS_CORE_EXPORT EclipseShadowReceiver {
 public:
  EclipseShadowReceiver(std::shared_ptr<cs::core::Settings> settings,
      std::shared_ptr<core::SolarSystem> solarSystem, scene::CelestialObject const* shadowReceiver);

  bool        needsRecompilation() const;
  std::string getShaderSnippet() const;

  void init(VistaGLSLShader* shader, uint32_t textureOffset);

  void update(double time, scene::CelestialObserver const& observer);

  void preRender() const;

  void postRender() const;

 private:
  static constexpr size_t MAX_BODIES = 8;

  const std::shared_ptr<cs::core::Settings> mSettings;
  const std::shared_ptr<core::SolarSystem>  mSolarSystem;
  scene::CelestialObject const* const       mShadowReceiver;

  VistaGLSLShader* mShader        = nullptr;
  uint32_t         mTextureOffset = 0;

  mutable EclipseShadowMode mLastEclipseShadowMode = EclipseShadowMode::eNone;

  std::array<glm::vec4, MAX_BODIES> mOccluders{};

  std::vector<std::shared_ptr<graphics::EclipseShadowMap>> mShadowMaps;

  struct {
    int sun;
    int numOccluders;
    int occluders;
    int shadowMaps;
  } mUniforms;
};
} // namespace cs::core
#endif // CS_CORE_ECLIPSE_SHADOW_RECEIVER_HPP
