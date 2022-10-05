////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_TRAJECTORIES_SUN_FLARE_HPP
#define CSP_TRAJECTORIES_SUN_FLARE_HPP

#include "Plugin.hpp"

#include "../../../src/cs-scene/CelestialObject.hpp"

#include <VistaBase/VistaColor.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <glm/glm.hpp>

namespace cs::core {
class Settings;
} // namespace cs::core

namespace csp::trajectories {

/// Adds an artificial flare effect around the object. Only makes sense for stars, but if you want
/// you can make anything glow like a christmas light :D. The SunFlare is hidden when HDR rendering
/// is enabled.
class SunFlare : public IVistaOpenGLDraw {
 public:
  /// The color of the flare.
  cs::utils::Property<VistaColor> pColor = VistaColor(1, 1, 1);

  SunFlare(std::shared_ptr<cs::core::Settings> settings,
      std::shared_ptr<Plugin::Settings>        pluginSettings,
      std::shared_ptr<cs::core::SolarSystem>   solarSystem);

  SunFlare(SunFlare const& other) = delete;
  SunFlare(SunFlare&& other)      = default;

  SunFlare& operator=(SunFlare const& other) = delete;
  SunFlare& operator=(SunFlare&& other) = default;

  ~SunFlare() override;

  /// The flare is attached to this body.
  void               setObjectName(std::string objectName);
  std::string const& getObjectName() const;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<cs::core::Settings>    mSettings;
  std::shared_ptr<Plugin::Settings>      mPluginSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::string                            mObjectName;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  VistaGLSLShader mShader;

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

#endif // CSP_TRAJECTORIES_SUN_FLARE_HPP
