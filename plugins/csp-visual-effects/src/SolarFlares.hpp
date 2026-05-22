////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_EFFECTS_PLUGIN_SOLAR_FLARES_HPP
#define CSP_VISUAL_EFFECTS_PLUGIN_SOLAR_FLARES_HPP

#include "Plugin.hpp"

#include "../../../src/cs-scene/CelestialObject.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/glm.hpp>

namespace csp::visualeffects {

class SolarFlares : public IVistaOpenGLDraw {
 public:
  SolarFlares(
    std::shared_ptr<Plugin::Settings>     pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>    solarSystem,
    std::shared_ptr<cs::core::TimeControl>    timeControl
  );

  SolarFlares(SolarFlares const& other) = delete;
  SolarFlares(SolarFlares&& other)      = delete;

  SolarFlares& operator=(SolarFlares const& other) = delete;
  SolarFlares& operator=(SolarFlares&& other)      = delete;

  ~SolarFlares() override;

  // This is called by the Plugin.
  void update(double tTime);

  // The axis visualizes the orientation of this object.
  void setParentName(std::string objectName);
  std::string const& getParentName() const;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  void createShader();

  std::shared_ptr<Plugin::Settings> mPluginSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::TimeControl> mTimeControl;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  int mVertexCount;

  std::string mParentName;

  bool mPlayBackTimeSet = false;
  float mPlaybackStartTime = 0.0f;

  std::unique_ptr<VistaTexture> mNoiseTexture;

  std::unique_ptr<VistaGLSLShader>        mShader;
  std::unique_ptr<VistaVertexArrayObject> mVAO;
  std::unique_ptr<VistaBufferObject>      mVBO;

  struct {
    uint32_t time  = 0;
    uint32_t resolution  = 0;
    uint32_t noiseTexture  = 0;
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
  } mUniforms;
};

} // namespace csp::visualeffects

#endif // CSP_VISUAL_EFFECTS_PLUGIN_SOLAR_FLARES_HPP