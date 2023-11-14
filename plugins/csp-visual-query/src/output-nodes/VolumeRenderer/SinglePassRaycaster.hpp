////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_SINGLEPASSRAYCASTER_HPP
#define CSP_VISUAL_QUERY_SINGLEPASSRAYCASTER_HPP

#include "../../types/types.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

#include <memory>
#include <string>

namespace csp::visualquery {

class SinglePassRaycaster final : public IVistaOpenGLDraw {
 public:
  SinglePassRaycaster(std::shared_ptr<cs::core::SolarSystem> solarSystem,
      std::shared_ptr<cs::core::Settings>                    settings);

  ~SinglePassRaycaster() override;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

  void        setData(std::shared_ptr<Volume3D> const& image);
  void        setCenter(std::string center);
  std::string getCenter() const;

private:
  std::string                            mObjectName;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::Settings>    mSettings;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  /// Vista GLSL shader object used for rendering
  VistaGLSLShader mShader;

  VistaTexture mTexture;
  bool         mHasTexture;

  /// Lower Corner of the bounding volume for the planet.
  glm::vec3 mMinBounds;

  /// Upper Corner of the bounding volume for the planet.
  glm::vec3 mMaxBounds;

  csl::ogc::Bounds3D mBounds;

  bool mShaderDirty = true;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_SINGLEPASSRAYCASTER_HPP
