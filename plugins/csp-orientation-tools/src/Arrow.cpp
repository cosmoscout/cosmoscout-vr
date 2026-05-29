////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Arrow.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaGroupNode.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::orientationtools {

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* ARROW_SHADER_VERT = R"(
#version 330

// inputs
layout(location = 0) in vec3 inPosition;

// uniforms
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main()
{
    vec4 pos = uMatModelView * vec4(inPosition.xyz, 1);
    gl_Position = uMatProjection * pos;
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* ARROW_SHADER_FRAG = R"(
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
    std::shared_ptr<cs::graphics::ObjLoader>    arrowModel,
    const glm::dvec3                            rotAxis,
    const float                                 rotAngle,
    const glm::vec4&                            color,
    float                                       size
  ) :
      mPluginSettings(std::move(pluginSettings)),
      mSolarSystem(std::move(solarSystem)),
      mRotAxis(rotAxis),
      mRotAngle(rotAngle),
      mColor(color),
      mSize(size)
  {

    // Add to scenegraph.
    VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
        mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));
    logger().info("Added arrow to scene graph.");

    // Remember vertex count fo arrow model.
    mVertexCount = static_cast<int>(arrowModel->getVertices().get()->size() / 3);

    // Create VBO and VAO from given vertices.
    mVBO = std::make_unique<VistaBufferObject>();
    mVAO = std::make_unique<VistaVertexArrayObject>();

    mVAO->Bind();

    mVBO->Bind(GL_ARRAY_BUFFER);
    mVBO->BufferData(arrowModel->getVertices().get()->size() * sizeof(float), arrowModel->getVertices().get()->data(), GL_DYNAMIC_DRAW);

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

void Arrow::setParentName(std::string objectName) {
  mParentName = std::move(objectName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& Arrow::getParentName() const {
  return mParentName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Arrow::update(double tTime) {
  return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Arrow::Do() {
  auto parent = mSolarSystem->getObject(mParentName);

  if ((!parent || !parent->getIsOrbitVisible()) || (!mPluginSettings->mEnableArrows.get())) {
    return true;
  }

  // Create shader
  createShader();

  // Get distance to observer and compute scale.
  glm::dvec3 observerRelative = parent->getObserverRelativePosition();
  double distanceToObserver = glm::length(observerRelative);
  double scale = distanceToObserver * mSize;

  // Get observer relative transform and extract the upper left 3x3 matrix.
  auto matMV = parent->getObserverRelativeTransform();
  
  glm::vec3 xAxis = glm::vec3(matMV[0]);
  glm::vec3 yAxis = glm::vec3(matMV[1]);
  glm::vec3 zAxis = glm::vec3(matMV[2]);

  // Reset the scale of the upper left 3x3 matrix and apply own scale.
  xAxis = glm::normalize(xAxis) * static_cast<float>(scale);
  yAxis = glm::normalize(yAxis) * static_cast<float>(scale);
  zAxis = glm::normalize(zAxis) * static_cast<float>(scale);

  // Inject the changed uper left 3x3 matrix into the whole of the 4x4 matrix.
  matMV[0] = glm::vec4(xAxis, 0.0f);
  matMV[1] = glm::vec4(yAxis, 0.0f);
  matMV[2] = glm::vec4(zAxis, 0.0f);

  // Rotate arrow depending on the coordinate axis.
  matMV = glm::rotate(matMV, static_cast<double>(glm::radians(mRotAngle)), mRotAxis);

  // Get projection matrix.
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_LINE_BIT);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST); // Makes lines visible through walls.

  mShader->Bind();

  // Set uniforms
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(glm::highp_mat4(matMV)));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
  mShader->SetUniform(mUniforms.color, mColor[0], mColor[1], mColor[2], mColor[3]);

  mVAO->Bind();

  // Draw arrow
  glDrawArrays(GL_TRIANGLES, 0, mVertexCount);

  // Cleanup
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
    
  mShader->Release();
  mVAO->Release();

  glPopAttrib();
  
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Arrow::createShader() {
  mShader = std::make_unique<VistaGLSLShader>();

  std::string sVert(ARROW_SHADER_VERT);
  std::string sFrag(ARROW_SHADER_FRAG);

  mShader->InitVertexShaderFromString(sVert);
  mShader->InitFragmentShaderFromString(sFrag);
  mShader->Link();

  mUniforms.color       = mShader->GetUniformLocation("cColor");
  mUniforms.modelViewMatrix  = mShader->GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader->GetUniformLocation("uMatProjection");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Arrow::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

} // namespace csp::orientationtools