////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "BoxRenderer.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace csp::virtualsatellite {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* BoxRenderer::BOX_VERT = R"(
#version 330 core

layout (location = 0) in vec3 iCenter;

uniform mat4 uMVP;

void main() {
    gl_Position = uMVP * vec4(iCenter, 1.0);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* BoxRenderer::BOX_FRAG = R"(
#version 330 core

uniform vec4 uColor;

out vec4 FragColor;

void main() {
    FragColor = uColor;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

BoxRenderer::BoxRenderer(std::shared_ptr<Plugin::Settings> pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>                 solarSystem)
    : mPluginSettings(std::move(pluginSettings))
    , mSolarSystem(std::move(solarSystem)) {

  mShader.InitVertexShaderFromString(BOX_VERT);
  mShader.InitFragmentShaderFromString(BOX_FRAG);
  mShader.Link();

  mUniforms.mvp   = mShader.GetUniformLocation("uMVP");
  mUniforms.color = mShader.GetUniformLocation("uColor");

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems) - 1);

  // clang-format off
  constexpr std::array cubeVertices = {
    -0.5f, -0.5f, -0.5f, // Front bottom left
    0.5f, -0.5f, -0.5f,  // Front bottom right
    0.5f, 0.5f, -0.5f,   // Front top right
    -0.5f, 0.5f, -0.5f,  // Front top left

    -0.5f, -0.5f, 0.5f, // Back bottom left
    0.5f, -0.5f, 0.5f,  // Back bottom right
    0.5f, 0.5f, 0.5f,   // Back top right
    -0.5f, 0.5f, 0.5f,  // Back top left
  };

  constexpr std::array<uint32_t, 36> cubeIndices = {
    // Front face
    0, 1, 2, 2, 3, 0,
    // Back face
    4, 7, 6, 6, 5, 4,
    // Left face
    4, 0, 3, 3, 7, 4,
    // Right face
    1, 5, 6, 6, 2, 1,
    // Top face
    3, 2, 6, 6, 7, 3,
    // Bottom face
    4, 5, 1, 1, 0, 4
  };
  // clang-format on

  GLuint vbo;
  GLuint ebo;
  glGenVertexArrays(1, &mVAO);
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);

  glBindVertexArray(mVAO);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices.data(), GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cubeIndices), cubeIndices.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), static_cast<void*>(nullptr));
  glEnableVertexAttribArray(0);

  glBindVertexArray(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

BoxRenderer::~BoxRenderer() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void BoxRenderer::setObjectName(std::string objectName) {
  mObjectName = std::move(objectName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void BoxRenderer::setBoxes(std::vector<Box> const& boxes) {
  mSolidBoxes.clear();
  mTranslucentBoxes.clear();
  for (auto const& box : boxes) {
    if (box.color.a < 1.0f) {
      mTranslucentBoxes.push_back(box);
    } else {
      mSolidBoxes.push_back(box);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void BoxRenderer::drawBox(Box const& box, glm::mat4 const& mvp) const {
  glm::mat4 boxMatrix = glm::mat4(1.0f);
  boxMatrix           = glm::translate(boxMatrix, box.pos);
  boxMatrix           = boxMatrix * glm::mat4_cast(box.rot);
  boxMatrix           = glm::scale(boxMatrix, box.size);

  glm::mat4 boxMVP = mvp * boxMatrix;

  glUniformMatrix4fv(mUniforms.mvp, 1, GL_FALSE, glm::value_ptr(boxMVP));
  glUniform4fv(mUniforms.color, 1, glm::value_ptr(box.color));

  glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, nullptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool BoxRenderer::Do() {
  auto object = mSolarSystem->getObject(mObjectName);
  if (!object || !object->getIsOrbitVisible()) {
    return true;
  }

  cs::utils::FrameStats::ScopedTimer timer("Box of " + mObjectName);
  // get model view and projection matrices
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matMVP = glm::make_mat4(glMatP.data()) * glm::make_mat4(glMatMV.data()) * glm::mat4(object->getObserverRelativeTransform());

  mShader.Bind();

  glBindVertexArray(mVAO);

  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);

  size_t i = 0;
  for (const auto& box : mSolidBoxes) {
    glm::vec3 scale{1};
    if (i++ < 3) {
      scale = glm::vec3{100, 100, 1};
    }
    drawBox({box.pos, box.rot, box.size * scale, box.color}, matMVP);
  }

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  for (auto const& box : mTranslucentBoxes) {
    drawBox({box.pos, box.rot, box.size * 100.F, box.color}, matMVP);
  }

  glPopAttrib();

  glBindVertexArray(0);

  mShader.Release();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool BoxRenderer::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::virtualsatellite
