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
} // namespace cs::scene

namespace cs::graphics {
struct EclipseShadowMap;
}

namespace cs::core {

class SolarSystem;
class Settings;

/// There are multiple ways to compute the eclipse shadow. Which one is used depends on the settings
/// key mGraphics.pEclipseShadowMode.
/// eNone                No eclipse shadows at all.
/// eDebug               Draws the umbra, antumbra and penumbra in different colors.
/// eLinear              Use a linear falloff in the penumbra and a quatratic in the antumbra.
/// eSmoothstep          Use a smoothstep falloff in the penumbra and a quatratic in the antumbra.
/// eCircleIntersection  Use cirlce intersection math to compute the occluded fraction of the Sun.
/// eTexture             Retrieve the amount of shadowing from a shadow-lookup texture.
/// eFastTexture         Like above, but with approaximations in the lookup-coordiante computation.
enum class EclipseShadowMode {
  eNone               = 0,
  eDebug              = 1,
  eLinear             = 2,
  eSmoothstep         = 3,
  eCircleIntersection = 4,
  eTexture            = 5,
  eFastTexture        = 6
};

/// Every object which should be able to receive eclipse shadows, should own an
/// EclipseShadowReceiver.
class CS_CORE_EXPORT EclipseShadowReceiver {
 public:
  EclipseShadowReceiver(std::shared_ptr<Settings> settings,
      std::shared_ptr<SolarSystem> solarSystem, scene::CelestialObject const* shadowReceiver);

  /// This will return true if mGraphics.pEclipseShadowMode has been changed since the last call to
  /// getShaderSnippet().
  bool needsRecompilation() const;

  /// Returns a GLSL snippet with the "vec3 getEclipseShadow(vec3 position)" method which should be
  /// included in the shader used for drawing the object. Before using the shader, you should check
  /// needsRecompilation() and if this returns true, you'll have to re-call this method.
  std::string getShaderSnippet() const;

  /// This should be called once the shader has been compiled. The textureOffset will be used for
  /// binding the eclipse shadow maps. There should be at least MAX_BODIES free texture units at
  /// this point.
  void init(VistaGLSLShader* shader, uint32_t textureOffset);

  /// This should be called once each frame.
  void update(double time, scene::CelestialObserver const& observer);

  /// This should be called before rendering the object. It will set all uniforms and bind the
  /// eclipse shadow maps.
  void preRender() const;

  /// This should be called after rendering the object. It will unbind all eclipse shadow maps
  void postRender() const;

 private:
  static constexpr size_t MAX_BODIES = 4;

  std::shared_ptr<Settings>     mSettings;
  std::shared_ptr<SolarSystem>  mSolarSystem;
  scene::CelestialObject const* mShadowReceiver;

  VistaGLSLShader* mShader        = nullptr;
  uint32_t         mTextureOffset = 0;

  std::array<glm::vec4, MAX_BODIES>                        mOccluders{};
  std::vector<std::shared_ptr<graphics::EclipseShadowMap>> mShadowMaps;

  mutable EclipseShadowMode mLastEclipseShadowMode = EclipseShadowMode::eNone;

  struct {
    int sun;
    int numOccluders;
    int occluders;
    int shadowMaps;
  } mUniforms;
};
} // namespace cs::core
#endif // CS_CORE_ECLIPSE_SHADOW_RECEIVER_HPP
