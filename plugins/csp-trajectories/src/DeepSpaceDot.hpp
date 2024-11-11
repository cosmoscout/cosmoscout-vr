////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_TRAJECTORIES_DEEP_SPACE_DOT_HPP
#define CSP_TRAJECTORIES_DEEP_SPACE_DOT_HPP

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaBase/VistaColor.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <glm/glm.hpp>

namespace csp::trajectories {

/// A deep space dot is a simple marker indicating the position of an object, when it is too
/// small to be seen. It can be used as a HUD element (a colorful circle which is drawn after tone
/// mapping) or as a flare (a bright glowing circle which is drawn before tone mapping).
class DeepSpaceDot : public IVistaOpenGLDraw {
 public:
  /// The mode of the marker.
  enum class Mode {
    /// In this mode, the marker is drawn after tone mapping.
    eMarker,

    /// In this mode, the marker is drawn before the planets and moons. A exponential glow effect
    /// is applied to the marker. This is useful to add an artificial glow to an object in non-HDR
    /// mode.
    eLDRFlare,

    /// In this mode, the marker is drawn after the planets. It is a simple circle with the given
    /// luminance. Thanks to the HDR rendering and the camera glow effect, the luminance will spread
    /// out and result in a realistic glowing effect.
    eHDRFlare,
  };

  /// Shows or hides the marker.
  cs::utils::Property<bool> pVisible = true;

  /// The mode of the marker.
  cs::utils::Property<Mode> pMode = Mode::eMarker;

  /// The color of the marker. In eHDRFlare mode, this corresponds to the average albedo of the
  /// object.
  cs::utils::Property<VistaColor> pColor = VistaColor(1, 1, 1, 1);

  /// This is used as a multiplier for the color. In eMarker and eLDRFlare mode, this should be
  /// 1.0. In eHDRFlare mode, this should be the luminance of the object.
  cs::utils::Property<float> pLuminance = 1.F;

  /// The solid angle of the marker in steradians.
  cs::utils::Property<float> pSolidAngle = 0.0001F;

  DeepSpaceDot(std::shared_ptr<cs::core::SolarSystem> solarSystem);

  DeepSpaceDot(DeepSpaceDot const& other) = delete;
  DeepSpaceDot(DeepSpaceDot&& other)      = default;

  DeepSpaceDot& operator=(DeepSpaceDot const& other) = delete;
  DeepSpaceDot& operator=(DeepSpaceDot&& other)      = default;

  ~DeepSpaceDot() override;

  void               setObjectName(std::string objectName);
  std::string const& getObjectName() const;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  VistaGLSLShader mShader;

  std::unique_ptr<VistaOpenGLNode>       mGLNode;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::string                            mObjectName;

  bool mShaderDirty = true;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t color            = 0;
    uint32_t solidAngle       = 0;
  } mUniforms;

  static const char* QUAD_VERT;
  static const char* QUAD_FRAG;
};

} // namespace csp::trajectories

#endif // CSP_TRAJECTORIES_DEEP_SPACE_DOT_HPP
