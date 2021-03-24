////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_RINGS_RING_HPP
#define CSP_RINGS_RING_HPP

#include "Plugin.hpp"

#include "../../../src/cs-scene/CelestialObject.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

namespace cs::core {
class SolarSystem;
} // namespace cs::core

namespace csp::rings {

/// A single planetary ring. It renders around the SPICE frame's center The texture should be an
/// RGBA cross-section of the ring, the inner and outer radius specify the distance from the
/// planet's center respectively.
class Ring : public cs::scene::CelestialObject, public IVistaOpenGLDraw {
 public:
  Ring(std::shared_ptr<cs::core::Settings>   settings,
      std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string const& anchorName);

  Ring(Ring const& other) = delete;
  Ring(Ring&& other)      = default;

  Ring& operator=(Ring const& other) = delete;
  Ring& operator=(Ring&& other) = delete;

  ~Ring() override;

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::Ring const& settings);

  /// The sun object is used for lighting computation.
  void setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun);

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<cs::core::Settings>               mSettings;
  std::shared_ptr<cs::core::SolarSystem>            mSolarSystem;
  std::shared_ptr<const cs::scene::CelestialObject> mSun;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  Plugin::Settings::Ring        mRingSettings;
  std::unique_ptr<VistaTexture> mTexture;
  VistaGLSLShader               mShader;
  VistaVertexArrayObject        mSphereVAO;
  VistaBufferObject             mSphereVBO;

  bool mShaderDirty         = true;
  int  mEnableHDRConnection = -1;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t surfaceTexture   = 0;
    uint32_t radii            = 0;
    uint32_t farClip          = 0;
    uint32_t sunIlluminance   = 0;
  } mUniforms;

  static const char* SPHERE_VERT;
  static const char* SPHERE_FRAG;
};
} // namespace csp::rings

#endif // CSP_RINGS_RING_HPP
