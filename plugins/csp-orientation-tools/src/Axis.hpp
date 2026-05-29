////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ORIENTATION_TOOLS_AXIS_HPP
#define CSP_ORIENTATION_TOOLS_AXIS_HPP

#include "Plugin.hpp"

#include "../../../src/cs-scene/CelestialObject.hpp"
#include "../../../src/cs-graphics/ObjLoader.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/glm.hpp>

namespace csp::orientationtools {

class Axis : public IVistaOpenGLDraw {
 public:
  Axis(
    std::shared_ptr<Plugin::Settings>     pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>    solarSystem,
    std::shared_ptr<cs::graphics::ObjLoader>  axisModel,
    const glm::dvec3                          rotAxis,
    const float                               rotAngle,
    const glm::vec4&                          color,
    float                                     size
  );

  Axis(Axis const& other) = delete;
  Axis(Axis&& other)      = delete;

  Axis& operator=(Axis const& other) = delete;
  Axis& operator=(Axis&& other)      = delete;

  ~Axis() override;

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

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  int mVertexCount;

  glm::vec4 mColor;
  float mSize;
  
  glm::dvec3 mRotAxis;
  float mRotAngle;

  std::string mParentName;

  std::unique_ptr<VistaGLSLShader>        mShader;
  std::unique_ptr<VistaVertexArrayObject> mVAO;
  std::unique_ptr<VistaBufferObject>      mVBO;

  struct {
    uint32_t color = 0;
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
  } mUniforms;
};

} // namespace csp::orientationtools

#endif // CSP_ORIENTATION_TOOLS_AXIS_HPP