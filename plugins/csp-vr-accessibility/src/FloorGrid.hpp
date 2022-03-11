////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VR_ACCESSIBILITY_FLOORGRID_HPP
#define CSP_VR_ACCESSIBILITY_FLOORGRID_HPP

#include "Plugin.hpp"

#include "../../../src/cs-scene/CelestialObject.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

namespace cs::core {
class Settings;
class SolarSystem;
} // namespace cs::core

namespace csp::vraccessibility {

/// The floor grid renders as a texturised quad below the observer.
/// The offset between the observer and the grid is set by a transformation node in the scenegraph.
/// The settings can be changed at runtime and determine the color and perceived size of the grid.
class FloorGrid : public IVistaOpenGLDraw {
 public:
  FloorGrid(
      std::shared_ptr<cs::core::SolarSystem> solarSystem, Plugin::Settings::Grid& gridSettings);

  FloorGrid(FloorGrid const& other) = delete;
  FloorGrid(FloorGrid&& other)      = default;

  FloorGrid& operator=(FloorGrid const& other) = delete;
  FloorGrid& operator=(FloorGrid&& other) = delete;

  ~FloorGrid();

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::Grid& gridSettings);

  /// Updates the offset of the grid according to the current settings
  void update();

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<cs::core::Settings>    mSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;

  std::unique_ptr<VistaTransformNode> mOffsetNode;
  std::unique_ptr<VistaOpenGLNode>    mGLNode;

  Plugin::Settings::Grid&       mGridSettings;
  std::unique_ptr<VistaTexture> mTexture;
  VistaGLSLShader               mShader;
  VistaVertexArrayObject        mVAO;
  VistaBufferObject             mVBO;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t texture          = 0;
    uint32_t extent           = 0;
    uint32_t size             = 0;
    uint32_t farClip          = 0;
    uint32_t alpha            = 0;
    uint32_t color            = 0;
  } mUniforms;

  static const char* VERT_SHADER;
  static const char* FRAG_SHADER;
}; // class FloorGrid
} // namespace csp::vraccessibility

#endif // CSP_VR_ACCESSIBILITY_FLOORGRID_HPP
