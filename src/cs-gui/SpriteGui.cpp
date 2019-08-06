////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SpriteGui.hpp"

#include "GuiArea.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaMath/VistaGeometries.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

namespace cs::gui {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string QUAD_VERT = R"(
#version 400 compatibility                                                 
                                                                            
vec2 positions[4] = vec2[](
    vec2(-0.5, -0.5),
    vec2(0.5, -0.5),
    vec2(-0.5, 0.5),
    vec2(0.5, 0.5)
);                               

uniform sampler2D uTexture;                                                
uniform vec2 uViewPort;                                                

out vec2 vTexCoords;                                                            
out vec3 vPosition;
                                                                           
void main()                                                                
{                       
  vTexCoords = vec2(positions[gl_VertexID].x, -positions[gl_VertexID].y) + 0.5;
  vec4 position = gl_ModelViewProjectionMatrix * vec4(0, 0, 0, 1);                                   

  position /= position.w;

  position.xy = (position.xy * uViewPort + 2.0 * positions[gl_VertexID] * textureSize(uTexture, 0)) / uViewPort;

  vPosition = (gl_ModelViewMatrix * vec4(0, 0, 0, 1)).xyz;
  gl_Position = position;                                   
}                                                                 
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string QUAD_FRAG = R"(
#version 400 compatibility                                                 
                                                                           
in vec2 vTexCoords;                                                             
in vec3 vPosition;

uniform sampler2D uTexture;                                                
uniform float uFarClip;                                                
                                                                           
layout(location = 0) out vec4 vOutColor;                                   
                                                                           
void main()                                                                
{                                                                                                    
  vOutColor = texture(uTexture, vTexCoords); 
  if (vOutColor.a == 0.0) discard; 

  vOutColor.rgb /= vOutColor.a;

  #if USE_LINEARDEPTHBUFFER
    // write linear depth
    gl_FragDepth = length(vPosition) / uFarClip;
  #endif
}  
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

SpriteGui::SpriteGui(std::string const& url, int width, int height)
    : WebView(url, width, height)
    , mTexture(new VistaTexture(GL_TEXTURE_2D))
    , mShader(new VistaGLSLShader()) {
  setDrawCallback([this](DrawEvent const& event) { updateTexture(event); });

  mTexture->SetWrapS(GL_CLAMP_TO_EDGE);
  mTexture->SetWrapT(GL_CLAMP_TO_EDGE);
  mTexture->UploadTexture(getWidth(), getHeight(), nullptr, false);
  mTexture->Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SpriteGui::~SpriteGui() {
  delete mTexture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SpriteGui::updateTexture(DrawEvent const& event) {
  mTexture->Bind();

  if (event.mResized) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, event.mWidth, event.mHeight, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, event.mData);
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, event.mX, event.mY, event.mWidth, event.mHeight, GL_BGRA,
        GL_UNSIGNED_BYTE, event.mData);
  }

  mTexture->Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SpriteGui::setDepthOffset(float offset) {
  mDepthOffset = offset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float SpriteGui::getDepthOffset() const {
  return mDepthOffset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SpriteGui::getUseLinearDepthBuffer() const {
  return mUseLinearDepthBuffer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SpriteGui::setUseLinearDepthBuffer(bool bEnable) {
  mShaderDirty          = true;
  mUseLinearDepthBuffer = bEnable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaTexture* SpriteGui::getTexture() const {
  return mTexture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SpriteGui::Do() {
  if (mShaderDirty) {
    delete mShader;
    mShader = new VistaGLSLShader();

    std::string sVert(QUAD_VERT);
    std::string sFrag(QUAD_FRAG);

    utils::replaceString(sFrag, "USE_LINEARDEPTHBUFFER", mUseLinearDepthBuffer ? "1" : "0");

    mShader->InitVertexShaderFromString(sVert);
    mShader->InitFragmentShaderFromString(sFrag);
    mShader->Link();

    mShaderDirty = false;
  }

  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  mShader->Bind();

  if (mUseLinearDepthBuffer) {
    double near, far;
    GetVistaSystem()
        ->GetDisplayManager()
        ->GetCurrentRenderInfo()
        ->m_pViewport->GetProjection()
        ->GetProjectionProperties()
        ->GetClippingRange(near, far);
    mShader->SetUniform(mShader->GetUniformLocation("iFarClip"), (float)far);
  }

  GLint viewport[4];
  glGetIntegerv(GL_VIEWPORT, viewport);

  mTexture->Bind(GL_TEXTURE0);
  mShader->SetUniform(mShader->GetUniformLocation("uTexture"), 0);
  mShader->SetUniform(mShader->GetUniformLocation("uViewPort"), viewport[2], viewport[3]);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  mTexture->Unbind(GL_TEXTURE0);

  mShader->Release();

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SpriteGui::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float fMin[3] = {-1.f, -1.f, -0.000001f};
  float fMax[3] = {1.f, 1.f, 0.000001f};

  oBoundingBox.SetBounds(fMin, fMax);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
