////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MouseRay.hpp"

#include "../cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>

#include <glm/gtc/type_ptr.hpp>

#include <array>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::array VERTICES{-0.5F, -0.5F, 0.F, -0.5F, 0.5F, 0.F, 0.5F, 0.5F, 0.F, 0.5F, -0.5F, 0.F,
    -0.5F, -0.5F, -1.F, -0.5F, 0.5F, -1.F, 0.5F, 0.5F, -1.F, 0.5F, -0.5F, -1.F};

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::array INDICES{0U, 1U, 2U, 2U, 3U, 0U, 0U, 4U, 5U, 5U, 1U, 0U, 4U, 7U, 6U, 6U, 5U, 4U, 2U,
    6U, 7U, 7U, 3U, 2U, 1U, 5U, 6U, 6U, 2U, 1U, 0U, 3U, 7U, 7U, 4U, 0U};

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SHADER_VERT = R"(
#version 330

uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

// inputs
layout(location = 0) in vec3 iPosition;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;

void main()
{
    vTexCoords = iPosition.xy;
    vPosition   = (uMatModelView * vec4(iPosition, 1.0)).xyz;
    gl_Position =  uMatProjection * vec4(vPosition, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SHADER_FRAG = R"(
#version 330

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

uniform float uFarClip;

// outputs
layout(location = 0) out vec4 oColor;

void main()
{
    oColor = vec4(1, 1, 1, 0.3);

    gl_FragDepth = length(vPosition) / uFarClip;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

MouseRay::MouseRay() {
  auto*               sceneGraph = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  VistaTransformNode* intentionNode =
      dynamic_cast<VistaTransformNode*>(sceneGraph->GetNode("SELECTION_NODE"));

  mRayTransform.reset(sceneGraph->NewTransformNode(intentionNode));
  mRayTransform->SetScale(0.001F, 0.001F, 30.F);

  mMouseRayNode.reset(sceneGraph->NewOpenGLNode(mRayTransform.get(), this));

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      intentionNode, static_cast<int>(cs::utils::DrawOrder::eRay));

  // Create box shader.
  mShader.InitVertexShaderFromString(SHADER_VERT);
  mShader.InitFragmentShaderFromString(SHADER_FRAG);
  mShader.Link();

  // Create box geometry.
  mRayVAO.Bind();

  mRayVBO.Bind(GL_ARRAY_BUFFER);
  mRayVBO.BufferData(VERTICES.size() * sizeof(float), VERTICES.data(), GL_STATIC_DRAW);

  mRayIBO.Bind(GL_ELEMENT_ARRAY_BUFFER);
  mRayIBO.BufferData(INDICES.size() * sizeof(unsigned), INDICES.data(), GL_STATIC_DRAW);

  mRayVAO.EnableAttributeArray(0);
  mRayVAO.SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0, &mRayVBO);

  mRayVAO.Release();
  mRayIBO.Release();
  mRayVBO.Release();

  mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
  mUniforms.farClip          = mShader.GetUniformLocation("uFarClip");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool MouseRay::Do() {
  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);
  glEnable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);

  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  mShader.Bind();
  mRayVAO.Bind();
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glMatMV.data());
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());

  mShader.SetUniform(mUniforms.farClip, utils::getCurrentFarClipDistance());

  glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(INDICES.size()), GL_UNSIGNED_INT, nullptr);
  mRayVAO.Release();
  mShader.Release();

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool MouseRay::GetBoundingBox(VistaBoundingBox& bb) {
  std::array fMin{-0.5F, -0.5F, -0.1F};
  std::array fMax{0.5F, 0.5F, 0.0F};

  bb.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
