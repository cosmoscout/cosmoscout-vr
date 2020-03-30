////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MouseRay.hpp"

#include "../cs-utils/utils.hpp"

#include <VistaMath/VistaBoundingBox.h>

#include <glm/gtc/type_ptr.hpp>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::vector<float> VERTICES = {-0.5f, -0.5f, 0.f, -0.5f, 0.5f, 0.f, 0.5f, 0.5f, 0.f, 0.5f,
    -0.5f, 0.f, -0.5f, -0.5f, -1.f, -0.5f, 0.5f, -1.f, 0.5f, 0.5f, -1.f, 0.5f, -0.5f, -1.f};

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::vector<unsigned> INDICES = {0, 1, 2, 2, 3, 0, 0, 4, 5, 5, 1, 0, 4, 7, 6, 6, 5, 4, 2, 6,
    7, 7, 3, 2, 1, 5, 6, 6, 2, 1, 0, 3, 7, 7, 4, 0};

////////////////////////////////////////////////////////////////////////////////////////////////////

static const std::string SHADER_VERT = R"(
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

static const std::string SHADER_FRAG = R"(
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool MouseRay::Do() {
  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);
  glEnable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);

  GLfloat glMatMV[16], glMatP[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, &glMatMV[0]);
  glGetFloatv(GL_PROJECTION_MATRIX, &glMatP[0]);

  mShader.Bind();
  mRayVAO.Bind();
  glUniformMatrix4fv(mShader.GetUniformLocation("uMatModelView"), 1, GL_FALSE, glMatMV);
  glUniformMatrix4fv(mShader.GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP);

  mShader.SetUniform(mShader.GetUniformLocation("uFarClip"), utils::getCurrentFarClipDistance());

  glDrawElements(GL_TRIANGLES, (GLsizei)INDICES.size(), GL_UNSIGNED_INT, nullptr);
  mRayVAO.Release();
  mShader.Release();

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool MouseRay::GetBoundingBox(VistaBoundingBox& bb) {
  float fMin[3] = {-0.5f, -0.5f, -0.1f};
  float fMax[3] = {0.5f, 0.5f, 0.0f};

  bb.SetBounds(fMin, fMax);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
