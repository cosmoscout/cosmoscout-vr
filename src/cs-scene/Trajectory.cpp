////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Trajectory.hpp"

#include "../cs-utils/utils.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

namespace cs::scene {

////////////////////////////////////////////////////////////////////////////////////////////////////

static const std::string SHADER_VERT = R"(
#version 330

// inputs
layout(location = 0) in vec3 inPosition;
layout(location = 1) in float inAge;

// uniforms
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

// outputs
out float fAge;
out vec4 vPosition;

void main()
{
    fAge = inAge;
    vPosition = uMatModelView * vec4(inPosition.xyz, 1);
    gl_Position = uMatProjection * vPosition;

  #if USE_LINEARDEPTHBUFFER
    gl_Position.z = 0;
  #else
    if (gl_Position.w > 0) {
      gl_Position /= gl_Position.w;
      if (gl_Position.z >= 1) {
        gl_Position.z = 0.999999;
      }
    }
  #endif
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

static const std::string SHADER_FRAG = R"(
#version 330

// inputs
in float fAge;
in vec4 vPosition;

// uniforms
uniform vec4 cStartColor;
uniform vec4 cEndColor;
uniform float fFarClip;

// outputs
layout(location = 0) out vec4 vOutColor;

void main()
{
  if (fAge < 0.0 || fAge > 1.0) discard;
  vOutColor = mix(cStartColor, cEndColor, fAge);

  #if USE_LINEARDEPTHBUFFER
    // write linear depth
    gl_FragDepth = length(vPosition.xyz) / fFarClip;
  #endif
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

Trajectory::Trajectory()
    : mShader(nullptr)
    , mVAO(nullptr)
    , mVBO(nullptr)
    , mMaxAge(100000.f)
    , mStartColor(1.f, 1.f, 1.f, 1.f)
    , mEndColor(1.f, 1.f, 1.f, 0.f)
    , mWidth(2.f)
    , mPointCount(0) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Trajectory::~Trajectory() {
  delete mShader;
  delete mVAO;
  delete mVBO;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::upload(glm::dmat4 const& relativeTransform, double dTime,
    std::vector<glm::dvec4> const& vPoints, glm::dvec3 const& vTip, int startIndex) {
  if (!vPoints.empty()) {
    // transform all points to observer centric coordinates
    std::vector<glm::vec4> points(vPoints.size());

    for (size_t i(0); i < vPoints.size(); ++i) {
      int ringbufferIndex = (i + startIndex) % (int)vPoints.size();

      glm::dvec4 const& curr = vPoints[ringbufferIndex];

      glm::dvec4 pos(curr.x, curr.y, curr.z, 1.0);
      auto       age = (float)((dTime - curr.w) / mMaxAge);

      if (curr.w >= dTime) {
        pos = glm::dvec4(vTip, 1.0);
        age = 0.f;
      }

      pos = relativeTransform * pos;

      points[i] = glm::vec4(pos.x, pos.y, pos.z, age);
    }

    if (mPointCount != vPoints.size()) {
      delete mVBO;
      delete mVAO;

      mVBO = new VistaBufferObject();
      mVAO = new VistaVertexArrayObject();

      mVAO->Bind();

      mVBO->Bind(GL_ARRAY_BUFFER);
      mVBO->BufferData(points.size() * sizeof(glm::vec4), points.data(), GL_DYNAMIC_DRAW);

      // positions
      mVAO->EnableAttributeArray(0);
      mVAO->SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), 0, mVBO);

      // ages
      mVAO->EnableAttributeArray(1);
      mVAO->SpecifyAttributeArrayFloat(
          1, 1, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), sizeof(glm::vec3), mVBO);

      mVAO->Release();
      mVBO->Release();
    } else {
      mVBO->Bind(GL_ARRAY_BUFFER);
      mVBO->BufferSubData(0, points.size() * sizeof(glm::vec4), points.data());
      mVBO->Release();
    }

    mPointCount = (int)vPoints.size();
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

    mVAO->Bind();
    mShader->Bind();

    if (mUseLinearDepthBuffer) {
      mShader->SetUniform(
          mShader->GetUniformLocation("fFarClip"), utils::getCurrentFarClipDistance());
    }

    mShader->SetUniform(mShader->GetUniformLocation("cStartColor"), mStartColor[0], mStartColor[1],
        mStartColor[2], mStartColor[3]);
    mShader->SetUniform(mShader->GetUniformLocation("cEndColor"), mEndColor[0], mEndColor[1],
        mEndColor[2], mEndColor[3]);

    // get modelview and projection matrices
    GLfloat glMatMV[16], glMatP[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, &glMatMV[0]);
    glGetFloatv(GL_PROJECTION_MATRIX, &glMatP[0]);
    glUniformMatrix4fv(mShader->GetUniformLocation("uMatModelView"), 1, GL_FALSE, glMatMV);
    glUniformMatrix4fv(mShader->GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP);

    glLineWidth(mWidth);

    uint32_t amountNoDepth = mPointCount / 2;

    glDepthMask(GL_FALSE);
    glDrawArrays(GL_LINE_STRIP, 0, amountNoDepth + 1);
    glDepthMask(GL_TRUE);

    glDrawArrays(GL_LINE_STRIP, amountNoDepth, mPointCount - amountNoDepth);

    mShader->Release();
    mVAO->Release();

    glPopAttrib();
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Trajectory::GetBoundingBox(VistaBoundingBox&) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::createShader() {
  delete mShader;
  mShader = new VistaGLSLShader();

  std::string sVert(SHADER_VERT);
  std::string sFrag(SHADER_FRAG);

  utils::replaceString(sFrag, "USE_LINEARDEPTHBUFFER", mUseLinearDepthBuffer ? "1" : "0");
  utils::replaceString(sVert, "USE_LINEARDEPTHBUFFER", mUseLinearDepthBuffer ? "1" : "0");

  mShader->InitVertexShaderFromString(sVert);
  mShader->InitFragmentShaderFromString(sFrag);
  mShader->Link();
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

bool Trajectory::getUseLinearDepthBuffer() const {
  return mUseLinearDepthBuffer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setUseLinearDepthBuffer(bool bEnable) {
  mShaderDirty          = true;
  mUseLinearDepthBuffer = bEnable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::scene
