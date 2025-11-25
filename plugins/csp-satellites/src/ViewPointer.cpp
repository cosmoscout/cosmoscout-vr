////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ViewPointer.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-scene/IntersectableObject.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "logger.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaDisplaySystem.h>
#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/InteractionManager/VistaUserPlatform.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <glm/gtc/type_ptr.hpp>

#include <utility>

namespace csp::satellites {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* ViewPointer::VERT_SHADER = R"(
#version 330

uniform mat4  uMatModelView;
uniform mat4  uMatProjection;

// inputs
layout(location = 0) in vec3 iPos;

void main() {
  vec4 vertPos = vec4(iPos, 1);
  vec3 pos    = (uMatModelView * vertPos).xyz;
  gl_Position = uMatProjection * vec4(pos, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* ViewPointer::FRAG_SHADER = R"(
#version 330

uniform float uAlpha;
uniform vec3  uCustomColor;

// outputs
layout(location = 0) out vec4 oColor;

void main() {
  oColor = vec4(uCustomColor, uAlpha);
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

ViewPointer::ViewPointer(
    std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string const& anchorName)
    : mSolarSystem(solarSystem)
    , mAnchorName(anchorName) {

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0, &mVBO);

  // Create shader
  mShader.InitVertexShaderFromString(VERT_SHADER);
  mShader.InitFragmentShaderFromString(FRAG_SHADER);
  mShader.Link();

  // Get Uniform Locations
  mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
  mUniforms.alpha            = mShader.GetUniformLocation("uAlpha");
  mUniforms.color            = mShader.GetUniformLocation("uCustomColor");

  // Add to scenegraph
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems) - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ViewPointer::~ViewPointer() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ViewPointer::configure(Plugin::Settings const& settings) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ViewPointer::update() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ViewPointer::Do() {
  cs::utils::FrameStats::ScopedTimer timer("Satellite-ViewPointer");

  mShader.Bind();

  auto       satelliteObject    = mSolarSystem->getObject(mAnchorName);
  auto       bodyObject         = mSolarSystem->getObject(mBodyName);
  auto       bodyIntersectable  = bodyObject->getIntersectableObject();
  glm::dmat4 satelliteTransform = satelliteObject->getObserverRelativeTransform();
  glm::dmat4 bodyTransform      = bodyObject->getObserverRelativeTransform();
  glm::dvec3 rayStart           = satelliteObject->getObserverRelativePosition();
  std::array<glm::dvec4, 4> rayDirs;
  rayDirs[0] = glm::dvec4(0.1, 0.1, 1, 0);
  rayDirs[1] = glm::dvec4(-0.1, 0.1, 1, 0);
  rayDirs[2] = glm::dvec4(-0.1, -0.1, 1, 0);
  rayDirs[3] = glm::dvec4(0.1, -0.1, 1, 0);
  std::vector<glm::vec3> vertices;
  for (glm::dvec4 rayDir : rayDirs) {
    rayDir = glm::normalize(satelliteTransform * rayDir);
    glm::dvec3 intersection;
    if (!bodyIntersectable->getIntersection(rayStart, rayDir, intersection)) {
      return true;
    }
    intersection = bodyTransform * glm::dvec4(intersection, 1.);
    vertices.emplace_back(rayStart);
    vertices.emplace_back(intersection);
  }
  for (int i = 0; i < 4; i++) {
    vertices.push_back(vertices[i * 2 + 1]);
    vertices.push_back(vertices[(i * 2 + 3) % 8]);
  }

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
  mVBO.Release();

  // Get modelview and projection matrices
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};

  glm::mat4 glMatM = glm::mat4(satelliteObject->getObserverRelativeTransform());
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  // Set uniforms
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glMatMV.data());
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
  mShader.SetUniform(mUniforms.alpha, 1.f);
  mShader.SetUniform(mUniforms.color, 0.1f, 0.8f, 0.1f);

  // Draw
  glPushAttrib(GL_ENABLE_BIT | GL_BLEND | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  glDepthMask(false);

  mVAO.Bind();

  // First we draw the grid with normal depth test.
  glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertices.size()));

  mVAO.Release();

  // Clean Up
  glPopAttrib();

  mShader.Release();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ViewPointer::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::satellites
