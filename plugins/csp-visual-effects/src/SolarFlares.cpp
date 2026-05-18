////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SolarFlares.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaGroupNode.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::visualeffects {

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SOLAR_FLARES_SHADER_VERT = R"(
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

static const char* SOLAR_FLARES_SHADER_FRAG = R"(
#version 330

// outputs
layout(location = 0) out vec4 vOutColor;

void main()
{
  vOutColor = vec4(1.0, 0.2, 0.2, 0.5);
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

SolarFlares::SolarFlares(std::shared_ptr<Plugin::Settings>  pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>      solarSystem
  ) :
      mPluginSettings(std::move(pluginSettings)),
      mSolarSystem(std::move(solarSystem))
  {

    // Add to scenegraph.
    VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
        mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems) + 1);
    logger().info("Added SolarFlares to scene graph.");

    // Solar flare will be depicted on a simple quad.
    std::vector<float> quadVertices = {
        // First Triangle
        -1.0f, -1.0f,  0.0f, // Bottom Left
         1.0f, -1.0f,  0.0f, // Bottom Right
        -1.0f,  1.0f,  0.0f, // Top Left

        // Second Triangle
         1.0f, -1.0f,  0.0f, // Bottom Right
         1.0f,  1.0f,  0.0f, // Top Right
        -1.0f,  1.0f,  0.0f  // Top Left
    };

    // Remember vertex count of quad for drawing.
    mVertexCount = 6;

    // Create VBO and VAO from given vertices.
    mVBO = std::make_unique<VistaBufferObject>();
    mVAO = std::make_unique<VistaVertexArrayObject>();

    mVAO->Bind();

    mVBO->Bind(GL_ARRAY_BUFFER);
    mVBO->BufferData(quadVertices.size() * sizeof(float), quadVertices.data(), GL_DYNAMIC_DRAW);

    mVAO->EnableAttributeArray(0);
    mVAO->SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0, mVBO.get());

    mVAO->Release();
    mVBO->Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SolarFlares::~SolarFlares() {
    VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarFlares::setParentName(std::string objectName) {
  mParentName = std::move(objectName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& SolarFlares::getParentName() const {
  return mParentName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarFlares::update(double tTime) {
  return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SolarFlares::Do() {
  auto parent = mSolarSystem->getObject(mParentName);

  if ((!parent || !parent->getIsBodyVisible())) {
    return true;
  }

  // Create shader
  createShader();

  // Get observer relative transform and extract the upper left 3x3 matrix.
  auto matMV = parent->getObserverRelativeTransform();

  // Remember the original position of the parent object.
  glm::dvec3 position = glm::dvec3(matMV[3]);

  // Remember the original scaling of the parent object.
  double scaleX = glm::length(glm::dvec3(matMV[0]));
  double scaleY = glm::length(glm::dvec3(matMV[1]));
  double scaleZ = glm::length(glm::dvec3(matMV[2]));

  // Set matrix to identity.
  matMV = glm::dmat4(1.0);

  // Size of the panel.
  double size = 1000.0 * 1000.0 * 1000.0; // TODO: Currently hardcoded size of panel.
  
  // Inject the original scaling together with the desired size of the panel.
  matMV[0][0] = scaleX * size;
  matMV[1][1] = scaleY * size;
  matMV[2][2] = scaleZ * size;

  // Inject the original position.
  matMV[3] = glm::dvec4(position, 1.0);

  // Get projection matrix.
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_LINE_BIT);
  glDisable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  mShader->Bind();

  // Set uniforms
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(glm::highp_mat4(matMV)));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());

  mVAO->Bind();

  // Draw panel with solar flare vfx.
  glDrawArrays(GL_TRIANGLES, 0, mVertexCount);

  // Cleanup
  glEnable(GL_CULL_FACE);
    
  mShader->Release();
  mVAO->Release();

  glPopAttrib();
  
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarFlares::createShader() {
  mShader = std::make_unique<VistaGLSLShader>();

  std::string sVert(SOLAR_FLARES_SHADER_VERT);
  std::string sFrag(SOLAR_FLARES_SHADER_FRAG);

  mShader->InitVertexShaderFromString(sVert);
  mShader->InitFragmentShaderFromString(sFrag);
  mShader->Link();

  mUniforms.modelViewMatrix  = mShader->GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader->GetUniformLocation("uMatProjection");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SolarFlares::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

} // namespace csp::visualeffects