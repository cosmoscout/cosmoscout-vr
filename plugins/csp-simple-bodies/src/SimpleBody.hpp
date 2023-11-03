////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_SIMPLE_BODIES_SIMPLE_PLANET_HPP
#define CSP_SIMPLE_BODIES_SIMPLE_PLANET_HPP

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include "../../../src/cs-core/EclipseShadowReceiver.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-scene/CelestialSurface.hpp"
#include "../../../src/cs-scene/IntersectableObject.hpp"

#include <memory>

#include "Plugin.hpp"

namespace cs::core {
class SolarSystem;
} // namespace cs::core

namespace csp::simplebodies {

/// This is just a sphere with a texture, attached to the given SPICE frame. The texture should be
/// in equirectangular projection.
class SimpleBody : public cs::scene::CelestialSurface,
                   public cs::scene::IntersectableObject,
                   public IVistaOpenGLDraw {
 public:
  SimpleBody(std::shared_ptr<cs::core::Settings> settings,
      std::shared_ptr<cs::core::SolarSystem>     solarSystem);

  SimpleBody(SimpleBody const& other) = delete;
  SimpleBody(SimpleBody&& other)      = default;

  SimpleBody& operator=(SimpleBody const& other) = delete;
  SimpleBody& operator=(SimpleBody&& other) = default;

  ~SimpleBody() override;

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::SimpleBody const& settings);

  /// The body is attached to this object.
  void               setObjectName(std::string objectName);
  std::string const& getObjectName() const;

  void update();

  /// Interface implementation of the IntersectableObject.
  bool getIntersection(
      glm::dvec3 const& rayOrigin, glm::dvec3 const& rayDir, glm::dvec3& pos) const override;

  /// Interface implementation of CelestialSurface.
  double getHeight(glm::dvec2 lngLat) const override;

  /// Interface implementation of IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<cs::core::Settings>    mSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;

  std::string mObjectName;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  Plugin::Settings::SimpleBody  mSimpleBodySettings;
  std::unique_ptr<VistaTexture> mTexture;
  VistaGLSLShader               mShader;
  VistaVertexArrayObject        mSphereVAO;
  VistaBufferObject             mSphereVBO;
  VistaBufferObject             mSphereIBO;

  std::unique_ptr<VistaTexture> mRingTexture;

  cs::core::EclipseShadowReceiver mEclipseShadowReceiver;

  bool mShaderDirty = true;

  int mEnableLightingConnection = -1;
  int mEnableHDRConnection      = -1;

  struct {
    uint32_t sunDirection      = 0;
    uint32_t sunIlluminance    = 0;
    uint32_t ambientBrightness = 0;
    uint32_t modelMatrix       = 0;
    uint32_t viewMatrix        = 0;
    uint32_t projectionMatrix  = 0;
    uint32_t surfaceTexture    = 0;
    uint32_t radii             = 0;
    uint32_t ringTexture       = 0;
    uint32_t ringRadii         = 0;
  } mUniforms;

  static const char* SPHERE_VERT;
  static const char* SPHERE_FRAG;
};

} // namespace csp::simplebodies

#endif // CSP_SIMPLE_BODIES_SIMPLE_PLANET_HPP
