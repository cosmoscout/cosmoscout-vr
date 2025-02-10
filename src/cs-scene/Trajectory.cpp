////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Trajectory.hpp"

#include "../cs-utils/utils.hpp"

#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>
#include <array>

namespace cs::scene {

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SHADER_VERT = R"(
#version 330

// inputs
layout(location = 0) in vec3 inPosition;
layout(location = 1) in float inAge;

// uniforms
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

// outputs
out float fAge;

void main()
{
    fAge = inAge;
    vec4 pos = uMatModelView * vec4(inPosition.xyz, 1);
    gl_Position = uMatProjection * pos;
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SHADER_FRAG = R"(
#version 330

// inputs
in float fAge;

// uniforms
uniform vec4 cStartColor;
uniform vec4 cEndColor;

// outputs
layout(location = 0) out vec4 vOutColor;

void main()
{
  if (fAge < 0.0 || fAge > 1.0) discard;
  vOutColor = mix(cStartColor, cEndColor, fAge);
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

Trajectory::Trajectory()
    : mStartColor(1.F, 1.F, 1.F, 1.F)
    , mEndColor(1.F, 1.F, 1.F, 0.F) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::upload(glm::dmat4 const& relativeTransform, double dTime,
    std::vector<glm::dvec4> const& vPoints, glm::dvec3 const& vTip, int startIndex) {
  if (!vPoints.empty()) {
    // transform all points to observer centric coordinates
    std::vector<glm::vec4> points(vPoints.size());

    for (size_t i(0); i < vPoints.size(); ++i) {
      int ringbufferIndex = (static_cast<int>(i) + startIndex) % static_cast<int>(vPoints.size());

      glm::dvec4 const& curr = vPoints[ringbufferIndex];

      glm::dvec4 pos(curr.x, curr.y, curr.z, 1.0);
      auto       age = static_cast<float>((dTime - curr.w) / mMaxAge);

      if (curr.w >= dTime) {
        pos = glm::dvec4(vTip, 1.0);
        age = 0.F;
      }

      pos = relativeTransform * pos;

      points[i] = glm::vec4(pos.x, pos.y, pos.z, age);
    }

    if (mPointCount != vPoints.size()) {
      mVBO = std::make_unique<VistaBufferObject>();
      mVAO = std::make_unique<VistaVertexArrayObject>();

      mVAO->Bind();

      mVBO->Bind(GL_ARRAY_BUFFER);
      mVBO->BufferData(points.size() * sizeof(glm::vec4), points.data(), GL_DYNAMIC_DRAW);

      // positions
      mVAO->EnableAttributeArray(0);
      mVAO->SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), 0, mVBO.get());

      // ages
      mVAO->EnableAttributeArray(1);
      mVAO->SpecifyAttributeArrayFloat(
          1, 1, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), sizeof(glm::vec3), mVBO.get());

      mVAO->Release();
      mVBO->Release();
    } else {
      mVBO->Bind(GL_ARRAY_BUFFER);
      mVBO->BufferSubData(0, points.size() * sizeof(glm::vec4), points.data());
      mVBO->Release();
    }

    mPointCount = static_cast<int>(vPoints.size());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Trajectory::Do() {
  if (mPointCount > 0 && mVAO) {
    if (mShaderDirty) {
      createShader();
      mShaderDirty = false;
    }

    glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_LINE_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glDepthMask(GL_FALSE);
    glLineWidth(mWidth);

    mVAO->Bind();
    mShader->Bind();

    mShader->SetUniform(
        mUniforms.startColor, mStartColor[0], mStartColor[1], mStartColor[2], mStartColor[3]);
    mShader->SetUniform(mUniforms.endColor, mEndColor[0], mEndColor[1], mEndColor[2], mEndColor[3]);

    // get modelview and projection matrices
    std::array<GLfloat, 16> glMatMV{};
    std::array<GLfloat, 16> glMatP{};
    glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
    glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
    glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glMatMV.data());
    glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());

    glDrawArrays(GL_LINE_STRIP, 0, mPointCount);

    mShader->Release();
    mVAO->Release();

    glPopAttrib();
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Trajectory::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::createShader() {
  mShader = std::make_unique<VistaGLSLShader>();

  std::string sVert(SHADER_VERT);
  std::string sFrag(SHADER_FRAG);

  mShader->InitVertexShaderFromString(sVert);
  mShader->InitFragmentShaderFromString(sFrag);
  mShader->Link();

  mUniforms.startColor       = mShader->GetUniformLocation("cStartColor");
  mUniforms.endColor         = mShader->GetUniformLocation("cEndColor");
  mUniforms.modelViewMatrix  = mShader->GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader->GetUniformLocation("uMatProjection");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double Trajectory::getMaxAge() const {
  return mMaxAge;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setMaxAge(double val) {
  mMaxAge = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec4 const& Trajectory::getStartColor() const {
  return mStartColor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setStartColor(glm::vec4 const& val) {
  mStartColor = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec4 const& Trajectory::getEndColor() const {
  return mEndColor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setEndColor(glm::vec4 const& val) {
  mEndColor = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float Trajectory::getWidth() const {
  return mWidth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setWidth(float val) {
  mWidth = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::scene