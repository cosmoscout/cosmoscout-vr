////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_ECLIPSE_SHADOW_RECEIVER_HPP
#define CS_CORE_ECLIPSE_SHADOW_RECEIVER_HPP

#include "GraphicsEngine.hpp"
#include "SolarSystem.hpp"

#include <VistaOGLExt/VistaGLSLShader.h>

#include <memory>
#include <vector>

namespace cs::scene {
class CelestialObject;
class CelestialBody;
} // namespace cs::scene

namespace cs::core {

class CS_CORE_EXPORT EclipseShadowReceiver {
 public:
  EclipseShadowReceiver(std::shared_ptr<cs::core::Settings> settings,
      std::shared_ptr<core::SolarSystem> solarSystem, scene::CelestialObject const* shadowReceiver);

  static std::string const& getShaderSnippet();

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

  std::array<glm::vec4, MAX_BODIES>                        mOccluders{};
  
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
