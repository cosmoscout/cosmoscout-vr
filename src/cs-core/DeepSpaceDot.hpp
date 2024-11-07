////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_DEEP_SPACE_DOT_HPP
#define CS_CORE_DEEP_SPACE_DOT_HPP

#include "SolarSystem.hpp"

#include <VistaBase/VistaColor.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <glm/glm.hpp>

namespace cs::core {

/// A deep space dot is a simple marker indicating the position of an object, when it is too
/// small to see.
class DeepSpaceDot : public IVistaOpenGLDraw {
 public:
  cs::utils::Property<VistaColor> pColor   = VistaColor(1, 1, 1); ///< The color of the marker.
  cs::utils::Property<bool>       pVisible = true;    ///< Whether the marker is visible.
  cs::utils::Property<float> pSolidAngle   = 0.0001F; ///< The marker's solid angle in steradians.
  cs::utils::Property<int> pSortKey = static_cast<int>(cs::utils::DrawOrder::eTransparentItems) - 1;

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

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t color            = 0;
    uint32_t solidAngle       = 0;
  } mUniforms;

  static const char* QUAD_VERT;
  static const char* QUAD_FRAG;
};

} // namespace cs::core

#endif // CSP_TRAJECTORIES_DEEP_SPACE_DOT_HPP
