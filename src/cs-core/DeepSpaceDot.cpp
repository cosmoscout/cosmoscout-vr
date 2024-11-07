////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "DeepSpaceDot.hpp"

#include "../cs-utils/FrameStats.hpp"
#include "../cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* DeepSpaceDot::QUAD_VERT = R"(
out vec2 vTexCoords;

uniform float uSolidAngle;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main()
{
    vec3 pos = (uMatModelView * vec4(0, 0, 0, 1)).xyz;

    float dist = length(pos);
    vec3 y = vec3(0, 1, 0);
    vec3 z = pos / dist;
    vec3 x = normalize(cross(z, y));
    y = normalize(cross(z, x));

    const float xy[2] = float[2](0.5, -0.5);
    const float PI = 3.14159265359;
  
    int i = gl_VertexID % 2;
    int j = gl_VertexID / 2;

    vTexCoords = vec2(xy[i], xy[j])*2;

    float diameter = 2.0 * sqrt(1 - pow(1-uSolidAngle/(2*PI), 2.0));
    float scale = dist * diameter;

    pos += (xy[i] * x + xy[j] * y) * scale;

    gl_Position = uMatProjection * vec4(pos, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* DeepSpaceDot::QUAD_FRAG = R"(
uniform vec3 uColor;

in vec2 vTexCoords;

layout(location = 0) out vec4 oColor;

void main()
{
    float dist = length(vTexCoords);

    // This is basically a cone from above. The average value is 1.0.
    float blob = clamp(1-dist, 0, 1) * 3;

  #ifdef ADDITIVE
    oColor = vec4(uColor * blob, 1.0);
  #else
    oColor = vec4(uColor, blob);
  #endif
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

DeepSpaceDot::DeepSpaceDot(std::shared_ptr<cs::core::SolarSystem> solarSystem)
    : mSolarSystem(std::move(solarSystem)) {

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));

  pSortKey.connectAndTouch(
      [this](int value) { VistaOpenSGMaterialTools::SetSortKeyOnSubtree(mGLNode.get(), value); });

  pAdditive.connectAndTouch([this](bool value) { mShaderDirty = true; });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DeepSpaceDot::~DeepSpaceDot() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DeepSpaceDot::setObjectName(std::string objectName) {
  mObjectName = std::move(objectName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& DeepSpaceDot::getObjectName() const {
  return mObjectName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DeepSpaceDot::Do() {
  if (!pVisible.get()) {
    return true;
  }

  auto object = mSolarSystem->getObject(mObjectName);
  if (!object || !object->getIsOrbitVisible()) {
    return true;
  }

  if (mShaderDirty) {
    std::string defines = "#version 330\n";

    if (pAdditive.get()) {
      defines += "#define ADDITIVE\n";
    }

    mShader = VistaGLSLShader();
    mShader.InitVertexShaderFromString(defines + QUAD_VERT);
    mShader.InitFragmentShaderFromString(defines + QUAD_FRAG);
    mShader.Link();

    mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
    mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
    mUniforms.color            = mShader.GetUniformLocation("uColor");
    mUniforms.solidAngle       = mShader.GetUniformLocation("uSolidAngle");

    mShaderDirty = false;
  }

  cs::utils::FrameStats::ScopedTimer timer("Dot of " + mObjectName);

  // get model view and projection matrices
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matMV = glm::make_mat4x4(glMatMV.data()) * glm::mat4(object->getObserverRelativeTransform());

  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);

  if (pAdditive.get()) {
    glBlendFunc(GL_ONE, GL_ONE);
  } else {
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }

  // draw simple dot
  mShader.Bind();
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
  mShader.SetUniform(mUniforms.color, pColor.get()[0], pColor.get()[1], pColor.get()[2]);
  mShader.SetUniform(mUniforms.solidAngle, pSolidAngle.get());
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  mShader.Release();

  glDisable(GL_BLEND);
  glDepthMask(GL_TRUE);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DeepSpaceDot::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
