////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_SIMPLE_BODIES_SIMPLE_PLANET_HPP
#define CSP_SIMPLE_BODIES_SIMPLE_PLANET_HPP

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-scene/CelestialBody.hpp"
#include "Plugin.hpp"

namespace cs::core {
class SolarSystem;
} // namespace cs::core

namespace csp::simplebodies {

/// This is just a sphere with a texture, attached to the given SPICE frame. The texture should be
/// in equirectangular projection.
class SimpleBody : public cs::scene::CelestialBody, public IVistaOpenGLDraw {
 public:
  SimpleBody(std::shared_ptr<cs::core::Settings> settings,
      std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string const& anchorName);

  SimpleBody(SimpleBody const& other) = delete;
  SimpleBody(SimpleBody&& other)      = default;

  SimpleBody& operator=(SimpleBody const& other) = delete;
  SimpleBody& operator=(SimpleBody&& other) = default;

  ~SimpleBody() override;

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::SimpleBody const& settings);

  /// The sun object is used for lighting computation.
  void setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun);

  /// Interface implementation of the IntersectableObject, which is a base class of
  /// CelestialBody.
  bool getIntersection(
      glm::dvec3 const& rayOrigin, glm::dvec3 const& rayDir, glm::dvec3& pos) const override;

  /// Interface implementation of CelestialBody.
  double getHeight(glm::dvec2 lngLat) const override;

  /// Interface implementation of IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<cs::core::Settings>               mSettings;
  std::shared_ptr<cs::core::SolarSystem>            mSolarSystem;
  std::shared_ptr<const cs::scene::CelestialObject> mSun;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  Plugin::Settings::SimpleBody  mSimpleBodySettings;
  std::unique_ptr<VistaTexture> mTexture;
  VistaGLSLShader               mShader;
  VistaVertexArrayObject        mSphereVAO;
  VistaBufferObject             mSphereVBO;
  VistaBufferObject             mSphereIBO;

  bool mShaderDirty              = true;
  int  mEnableLightingConnection = -1;
  int  mEnableHDRConnection      = -1;

  struct {
    uint32_t sunDirection      = 0;
    uint32_t sunIlluminance    = 0;
    uint32_t ambientBrightness = 0;
    uint32_t modelViewMatrix   = 0;
    uint32_t projectionMatrix  = 0;
    uint32_t surfaceTexture    = 0;
    uint32_t radii             = 0;
    uint32_t farClip           = 0;
  } mUniforms;

  static const char* SPHERE_VERT;
  static const char* SPHERE_FRAG;
};

} // namespace csp::simplebodies

#endif // CSP_SIMPLE_BODIES_SIMPLE_PLANET_HPP
