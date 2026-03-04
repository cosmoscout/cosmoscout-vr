////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_COORDINATE_ARROWS_ARROWS_HPP
#define CSP_COORDINATE_ARROWS_ARROWS_HPP

#include "Plugin.hpp"

#include "../../../src/cs-scene/CelestialObject.hpp"

#include <VistaBase/VistaColor.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <vector>

namespace csp::coordinatearrows {

class Arrows : public IVistaOpenGLDraw {
 public:
  Arrows(std::shared_ptr<Plugin::Settings> pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>   solarSystem);

  Arrows(Arrows const& other) = delete;
  Arrows(Arrows&& other)      = delete;

  Arrows& operator=(Arrows const& other) = delete;
  Arrows& operator=(Arrows&& other)      = delete;

  ~Arrows() override;

  // This is called by the Plugin.
  void update(double tTime);

  // The arrows visualize the orientation of this object.
  void setParentName(std::string objectName);
  std::string const& getParentName() const;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  void createShader();

  std::shared_ptr<Plugin::Settings> mPluginSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  std::string mParentName;

  std::unique_ptr<VistaGLSLShader>        mShader;
  std::unique_ptr<VistaVertexArrayObject> mVAO;
  std::unique_ptr<VistaBufferObject>      mVBO;

  glm::vec4 mColor;

  struct {
    uint32_t color = 0;
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
  } mUniforms;

  std::vector<glm::dvec4> mPointsXArrow;
  std::vector<glm::dvec4> mPointsYArrow;
  std::vector<glm::dvec4> mPointsZArrow;
  double mArrowLength = 1000;
  GLfloat mArrowWidth = 1000;
};

} // namespace csp::coordinatearrows

#endif // CSP_TRAJECTORIES_TRAJECTORY_HPP
