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
#define _USE_MATH_DEFINES
#include <math.h>

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

ViewPointer::ViewPointer(Plugin::Settings::Satellite const& config,
    std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string const& anchorName)
    : mSolarSystem(solarSystem)
    , mAnchorName(anchorName)
    , mFieldOfView(config.mFieldOfView.get()) {

  mVAO.Bind();
  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0, &mVBO);

  std::vector<unsigned> indicesLines     = {0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1};
  mIndexCountLines                       = static_cast<unsigned int>(indicesLines.size());
  std::vector<unsigned> indicesTriangles = {0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 1, 2, 3, 1, 4, 3};
  mIndexCountTriangles                   = static_cast<unsigned int>(indicesTriangles.size());

  mIBOTriangles.Bind(GL_ELEMENT_ARRAY_BUFFER);
  mIBOTriangles.BufferData(
      indicesTriangles.size() * sizeof(unsigned), indicesTriangles.data(), GL_STATIC_DRAW);
  mIBOLines.Bind(GL_ELEMENT_ARRAY_BUFFER);
  mIBOLines.BufferData(indicesLines.size() * sizeof(unsigned), indicesLines.data(), GL_STATIC_DRAW);

  mVAO.Release();
  mIBOLines.Release();

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

  config.mFieldOfView.connect([this](double val) { mFieldOfView = val; });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ViewPointer::~ViewPointer() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
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

  double angleRad = mFieldOfView / 180. * M_PI;
  double dx =
      std::sqrt(((1. / std::cos(angleRad / 2.)) * (1. / std::cos(angleRad / 2.)) - 1.) / 2.);

  std::array<glm::dvec4, 4> rayDirs;
  rayDirs[0] = glm::dvec4(dx, dx, 1, 0);
  rayDirs[1] = glm::dvec4(-dx, dx, 1, 0);
  rayDirs[2] = glm::dvec4(-dx, -dx, 1, 0);
  rayDirs[3] = glm::dvec4(dx, -dx, 1, 0);
  std::vector<glm::vec3> vertices;
  vertices.emplace_back(rayStart);
  for (int i = 0; i < 4; i++) {
    glm::dvec4 rayDir = rayDirs[i];
    rayDir            = glm::normalize(satelliteTransform * rayDir);
    glm::dvec3 intersection;
    if (!bodyIntersectable->getIntersection(rayStart, rayDir, intersection)) {
      intersection = rayStart + rayDir.xyz() * mLastDist[i];
    } else {
      intersection = bodyTransform * glm::dvec4(intersection, 1.);
      intersection = rayStart + (intersection - rayStart) * 0.999;
      mLastDist[i] = glm::length(intersection - rayStart);
    }
    vertices.emplace_back(intersection);
  }

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
  mVBO.Release();

  // Get modelview and projection matrices
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};

  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  // Set uniforms
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glMatMV.data());
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
  mShader.SetUniform(mUniforms.color, 0.1f, 0.8f, 0.1f);

  // Draw
  glPushAttrib(GL_ENABLE_BIT | GL_BLEND | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(false);

  mVAO.Bind();
  mShader.SetUniform(mUniforms.alpha, 1.f);
  mIBOLines.Bind(GL_ELEMENT_ARRAY_BUFFER);
  glDrawElements(GL_LINES, mIndexCountLines, GL_UNSIGNED_INT, nullptr);
  mShader.SetUniform(mUniforms.alpha, .1f);
  mIBOTriangles.Bind(GL_ELEMENT_ARRAY_BUFFER);
  glDrawElements(GL_TRIANGLES, mIndexCountTriangles, GL_UNSIGNED_INT, nullptr);
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
