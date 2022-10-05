////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_TRAJECTORIES_DEEP_SPACE_DOT_HPP
#define CSP_TRAJECTORIES_DEEP_SPACE_DOT_HPP

#include "Plugin.hpp"

#include "../../../src/cs-scene/CelestialObject.hpp"

#include <VistaBase/VistaColor.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <glm/glm.hpp>

namespace csp::trajectories {

/// A deep space dot is a simple marker indicating the position of an object, when it is too
/// small to see.
class DeepSpaceDot : public IVistaOpenGLDraw {
 public:
  cs::utils::Property<VistaColor> pColor = VistaColor(1, 1, 1); ///< The color of the marker.

  DeepSpaceDot(std::shared_ptr<Plugin::Settings> pluginSettings,
      std::shared_ptr<cs::core::SolarSystem>     solarSystem);

  DeepSpaceDot(DeepSpaceDot const& other) = delete;
  DeepSpaceDot(DeepSpaceDot&& other)      = default;

  DeepSpaceDot& operator=(DeepSpaceDot const& other) = delete;
  DeepSpaceDot& operator=(DeepSpaceDot&& other) = default;

  ~DeepSpaceDot() override;

  /// The dot is attached to this body.
  void               setObjectName(std::string objectName);
  std::string const& getObjectName() const;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<Plugin::Settings>      mPluginSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::string                            mObjectName;
  VistaGLSLShader                        mShader;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t color            = 0;
    uint32_t aspect           = 0;
  } mUniforms;

  static const char* QUAD_VERT;
  static const char* QUAD_FRAG;
};

} // namespace csp::trajectories

#endif // CSP_TRAJECTORIES_DEEP_SPACE_DOT_HPP
