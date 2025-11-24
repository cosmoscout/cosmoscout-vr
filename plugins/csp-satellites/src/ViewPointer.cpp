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

uniform mat4  uMatModel;
uniform mat4  uMatModelView;
uniform mat4  uMatProjection;
uniform vec3  uRayStart;
uniform vec3  uRayEnd;

// inputs
layout(location = 0) in vec2 iQuadPos;

// outputs
out vec2  vTexCoords;

void main() {
  vTexCoords  = iQuadPos;
  vec4 vertPos = vec4(gl_VertexID % 2 == 0 ? uRayStart : uRayEnd, 1);
  vec3 pos    = (uMatModelView * vertPos).xyz;
  gl_Position = uMatProjection * vec4(pos, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* ViewPointer::FRAG_SHADER = R"(
#version 330

// inputs
in vec2 vTexCoords;

// outputs
layout(location = 0) out vec4 oColor;

void main() {
  oColor = vec4(0.3, 0.5, 0.4, 1.);
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

ViewPointer::ViewPointer(
    std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string const& anchorName)
    : mSolarSystem(solarSystem)
    , mAnchorName(anchorName) {

  // Create initial Quad
  std::array<glm::vec2, 4> vertices{};
  vertices[0] = glm::vec2(-1.F, -1.F);
  vertices[1] = glm::vec2(1.F, -1.F);
  vertices[2] = glm::vec2(1.F, 1.F);
  vertices[3] = glm::vec2(-1.F, 1.F);

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(vertices.size() * sizeof(glm::vec2), vertices.data(), GL_STATIC_DRAW);
  mVBO.Release();

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0, &mVBO);

  // Create shader
  mShader.InitVertexShaderFromString(VERT_SHADER);
  mShader.InitFragmentShaderFromString(FRAG_SHADER);
  mShader.Link();

  // Get Uniform Locations
  mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
  mUniforms.modelMatrix      = mShader.GetUniformLocation("uMatModel");
  mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
  mUniforms.rayStart         = mShader.GetUniformLocation("uRayStart");
  mUniforms.rayEnd           = mShader.GetUniformLocation("uRayEnd");
  mUniforms.texture          = mShader.GetUniformLocation("uTexture");
  mUniforms.extent           = mShader.GetUniformLocation("uExtent");
  mUniforms.size             = mShader.GetUniformLocation("uSize");
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
  glm::dvec4 rayDir(0, 0, 1, 0);
  rayDir = glm::normalize(satelliteTransform * rayDir);
  glm::dvec3 intersection;
  if (!bodyIntersectable->getIntersection(rayStart, rayDir, intersection)) {
    return true;
  }
  intersection = bodyTransform * glm::dvec4(intersection, 1.);

  // Get modelview and projection matrices
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};

  glm::mat4 glMatM = glm::mat4(satelliteObject->getObserverRelativeTransform());
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  // Set uniforms
  glUniformMatrix4fv(mUniforms.modelMatrix, 1, GL_FALSE, glm::value_ptr(glMatM));
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glMatMV.data());
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
  glUniform3f(mUniforms.rayStart, static_cast<float>(rayStart[0]), static_cast<float>(rayStart[1]),
      static_cast<float>(rayStart[2]));
  glUniform3f(mUniforms.rayEnd, static_cast<float>(intersection[0]),
      static_cast<float>(intersection[1]), static_cast<float>(intersection[2]));
  mShader.SetUniform(mUniforms.texture, 0);

  // Draw
  glPushAttrib(GL_ENABLE_BIT | GL_BLEND | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_CULL_FACE);
  glDepthMask(false);

  mVAO.Bind();

  // First we draw the grid with normal depth test.
  glDrawArrays(GL_LINES, 0, 2);

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
