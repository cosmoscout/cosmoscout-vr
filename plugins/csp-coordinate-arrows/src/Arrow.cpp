////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Arrow.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::coordinatearrows {

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SHADER_VERT = R"(
#version 330

// inputs
layout(location = 0) in vec3 inPosition;

// uniforms
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;
uniform float uScaleFactor;

void main()
{
    vec3 stretchedArrow = vec3(inPosition.xyz) * uScaleFactor;
    vec4 pos = uMatModelView * vec4(stretchedArrow.xyz, 1);
    gl_Position = uMatProjection * pos;
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SHADER_FRAG = R"(
#version 330

// uniforms
uniform vec4 cColor;

// outputs
layout(location = 0) out vec4 vOutColor;

void main()
{
  vOutColor = cColor;
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

Arrow::Arrow(std::shared_ptr<Plugin::Settings>  pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>      solarSystem,
    const std::vector<float>&                         directionFromOrigin,
    const glm::vec4&                                  color
  ) :
      mPluginSettings(std::move(pluginSettings)),
      mSolarSystem(std::move(solarSystem)),
      mColor(color)
  {

    // Add to scenegraph.
    VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
    mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems) + 1);
    logger().info("Added arrow to scene graph.");

    // Create the vertices of the line by adding direction vertex to origin vertex.
    std::vector<float> lineVertices = {0.0f, 0.0f, 0.0f};
    lineVertices.reserve(lineVertices.size() + directionFromOrigin.size());
    lineVertices.insert(lineVertices.end(), directionFromOrigin.begin(), directionFromOrigin.end());

    mVBO = std::make_unique<VistaBufferObject>();
    mVAO = std::make_unique<VistaVertexArrayObject>();

    mVAO->Bind();

    mVBO->Bind(GL_ARRAY_BUFFER);
    mVBO->BufferData(sizeof(lineVertices), lineVertices.data(), GL_DYNAMIC_DRAW);

    mVAO->EnableAttributeArray(0);
    mVAO->SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0, mVBO.get());

    mVAO->Release();
    mVBO->Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Arrow::~Arrow() {
    VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Arrow::update(double tTime) {
    return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Arrow::setParentName(std::string objectName) {
  mParentName = std::move(objectName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& Arrow::getParentName() const {
  return mParentName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Arrow::Do() {
  auto parent = mSolarSystem->getObject(mParentName);

  if ((!parent || !parent->getIsBodyVisible()) || (!mPluginSettings->mEnableArrows.get())) {
    return true;
  }

  // logger().info("Drawing arrow.");
  // Create shader
  createShader();

  // Get modelview and projection matrices
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  // Make matrix relative to object
  auto matMV = glm::make_mat4x4(glMatMV.data()) * glm::mat4(parent->getObserverRelativeTransform());

  //glm::vec3 observerRelative = parent->getObserverRelativePosition();
  //float dist = glm::length(observerRelative);

  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_LINE_BIT);
  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_LINE_SMOOTH);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glDisable(GL_DEPTH_TEST);   // Makes lines visible through walls.
  glLineWidth(mArrowWidth);

  mShader->Bind();

  // Set uniforms
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
  mShader->SetUniform(mUniforms.color, mColor[0], mColor[1], mColor[2], mColor[3]);
  mShader->SetUniform(mUniforms.scaleFactor, 10.0f /*dist*/);

  //logger().info("Observer Distance: {} with vector {}, {}, {}", dist, observerRelative.x, observerRelative.y, observerRelative.z);

  mVAO->Bind();

  // Draw arrow
  glDrawArrays(GL_LINE_STRIP, 0, 2);

  // Cleanup
  glEnable(GL_DEPTH_TEST);
  mShader->Release();
  mVAO->Release();

  glPopAttrib();
  
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Arrow::createShader() {
  mShader = std::make_unique<VistaGLSLShader>();

  std::string sVert(SHADER_VERT);
  std::string sFrag(SHADER_FRAG);

  mShader->InitVertexShaderFromString(sVert);
  mShader->InitFragmentShaderFromString(sFrag);
  mShader->Link();

  mUniforms.color       = mShader->GetUniformLocation("cColor");
  mUniforms.modelViewMatrix  = mShader->GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader->GetUniformLocation("uMatProjection");
  mUniforms.scaleFactor = mShader->GetUniformLocation("uScaleFactor");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Arrow::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

} // namespace csp::coordinatearrows
