////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_COORDINATE_ARROWS_ARROW_HPP
#define CSP_COORDINATE_ARROWS_ARROW_HPP

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

class Arrow : public IVistaOpenGLDraw {
 public:
  Arrow(std::shared_ptr<Plugin::Settings>   pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>  solarSystem,
    std::vector<float>                      arrowVertices,
    const glm::dvec3                        rotAxis,
    const float                             rotAngle,
    const glm::vec4&                        color,
    float                                   width,
    float                                   size
  );

  Arrow(Arrow const& other) = delete;
  Arrow(Arrow&& other)      = delete;

  Arrow& operator=(Arrow const& other) = delete;
  Arrow& operator=(Arrow&& other)      = delete;

  ~Arrow() override;

  // This is called by the Plugin.
  void update(double tTime);

  // The arrow visualize the orientation of this object.
  void setParentName(std::string objectName);
  std::string const& getParentName() const;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  void createShader();

  std::shared_ptr<Plugin::Settings> mPluginSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  glm::vec4 mColor;
  float mWidth;
  float mSize;
  
  int mVertexCount;
  glm::dvec3 mRotAxis;
  float mRotAngle;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

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

} // namespace csp::coordinatearrows

#endif // CSP_TRAJECTORIES_TRAJECTORY_HPP
