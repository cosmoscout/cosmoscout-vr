////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "DeepSpaceDot.hpp"

#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace csp::trajectories {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* DeepSpaceDot::QUAD_VERT = R"(
out vec2 vTexCoords;

uniform float uSolidAngle;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main() {
  vec3 pos = (uMatModelView * vec4(0, 0, 0, 1)).xyz;

  float dist = length(pos);
  vec3 y = vec3(0, 1, 0);
  vec3 z = pos / dist;
  vec3 x = normalize(cross(z, y));
  y = normalize(cross(z, x));

  float drawDist = 0.9 * dist;

  const float offset[2] = float[2](0.5, -0.5);
  const float PI = 3.14159265359;

  int i = gl_VertexID % 2;
  int j = gl_VertexID / 2;

  vTexCoords = vec2(offset[i], offset[j]) * 2.0;

  float diameter = sqrt(uSolidAngle / (4 * PI)) * 4.0;

#if defined(FLARE_MODE)
  diameter *= 10.0;
#endif

  float scale = drawDist * diameter;

  pos += (offset[i] * x + offset[j] * y) * scale;

  gl_Position = uMatProjection * vec4(pos / dist * drawDist, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* DeepSpaceDot::QUAD_FRAG = R"(
uniform vec4 uColor;

in vec2 vTexCoords;

layout(location = 0) out vec4 oColor;

void main() {
  float dist = length(vTexCoords);

#if defined(CIRCLE_MODE)
  // Draw an pseudo-anti-aliased circle.
  float alpha = 1.0 - smoothstep(0.9, 1.0, dist);
  oColor = vec4(uColor.rgb, alpha * uColor.a);

#elif defined(HDR_BILLBOARD_MODE)
  // This is basically a cone from above. The average value is 1.0.
  float intensity = clamp(1.0 - dist, 0.0, 1.0) * 3.0;
  oColor = vec4(uColor.rgb * intensity, uColor.a);

#elif defined(FLARE_MODE)
  // The quad is drawn ten times larger than the object it represents. Hence we should make the
  // glow function equal to one at a distance of 0.1.
  float alpha = 1.0 - pow(clamp((dist - 0.1) / (1.0 - 0.1), 0.0, 1.0), 0.2);
  oColor = vec4(uColor.rgb, alpha * uColor.a);

#endif

}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

DeepSpaceDot::DeepSpaceDot(std::shared_ptr<cs::core::SolarSystem> solarSystem)
    : mSolarSystem(std::move(solarSystem)) {

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));

  pDrawOrder.connectAndTouch(
      [this](int order) { VistaOpenSGMaterialTools::SetSortKeyOnSubtree(mGLNode.get(), order); });

  pDrawOrder.connectAndTouch([this](int order) { mShaderDirty = true; });
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
  if (!object) {
    return true;
  }

  if (mShaderDirty) {
    std::string defines = "#version 330\n";

    if (pMode.get() == Mode::eSmoothCircle) {
      defines += "#define CIRCLE_MODE\n";
    } else if (pMode.get() == Mode::eHDRBillboard) {
      defines += "#define HDR_BILLBOARD_MODE\n";
    } else if (pMode.get() == Mode::eFlare) {
      defines += "#define FLARE_MODE\n";
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

  cs::utils::FrameStats::ScopedTimer timer("DeepSpaceDot for " + mObjectName);

  // get model view and projection matrices
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matMV = glm::make_mat4x4(glMatMV.data()) * glm::mat4(object->getObserverRelativeTransform());

  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);

  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // draw simple dot
  mShader.Bind();
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
  mShader.SetUniform(mUniforms.color, pLuminance.get() * pColor.get()[0],
      pLuminance.get() * pColor.get()[1], pLuminance.get() * pColor.get()[2], pColor.get()[3]);
  mShader.SetUniform(mUniforms.solidAngle, pSolidAngle.get());
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  mShader.Release();

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DeepSpaceDot::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::trajectories
