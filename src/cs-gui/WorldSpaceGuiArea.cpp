////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WorldSpaceGuiArea.hpp"

#include "../cs-utils/FrameTimings.hpp"
#include "GuiItem.hpp"

#include <VistaMath/VistaBoundingBox.h>
#include <VistaMath/VistaGeometries.h>

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string QUAD_VERT = R"(
vec2 positions[4] = vec2[](
    vec2(-0.5, -0.5),
    vec2(0.5, -0.5),
    vec2(-0.5, 0.5),
    vec2(0.5, 0.5)
);

uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

out vec2 vTexCoords;
out vec4 vPosition;

void main()
{
  vec2 p = positions[gl_VertexID];
  vTexCoords = vec2(p.x, -p.y) + 0.5;
  vPosition = uMatModelView * vec4(p, 0, 1);
  gl_Position = uMatProjection * vPosition;

  #ifdef USE_LINEARDEPTHBUFFER
    gl_Position.z = 0;
  #else
    if (gl_Position.w > 0) {
      gl_Position /= gl_Position.w;
      if (gl_Position.z >= 1) {
        gl_Position.z = 0.999999;
      }
    }
  #endif
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string QUAD_FRAG = R"(
in vec2 vTexCoords;                                                             
in vec4 vPosition;

uniform sampler2D iTexture;                                                
uniform float iFarClip;                                                
                                                                           
layout(location = 0) out vec4 vOutColor;                                   
                                                                           
void main()                                                                
{                                                                                                    
  vOutColor = texture(iTexture, vTexCoords); 
  if (vOutColor.a == 0.0) discard; 

  vOutColor.rgb /= vOutColor.a;

  #ifdef USE_LINEARDEPTHBUFFER
    // write linear depth
    gl_FragDepth = length(vPosition.xyz) / iFarClip;
  #endif
}  
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

namespace cs::gui {

////////////////////////////////////////////////////////////////////////////////////////////////////

WorldSpaceGuiArea::WorldSpaceGuiArea(int width, int height)
    : mShader(new VistaGLSLShader())
    , mWidth(width)
    , mHeight(height) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WorldSpaceGuiArea::~WorldSpaceGuiArea() {
  delete mShader;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WorldSpaceGuiArea::setWidth(int width) {
  mWidth = width;
  updateItems();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WorldSpaceGuiArea::setHeight(int height) {
  mHeight = height;
  updateItems();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int WorldSpaceGuiArea::getWidth() const {
  return mWidth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int WorldSpaceGuiArea::getHeight() const {
  return mHeight;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WorldSpaceGuiArea::setIgnoreDepth(bool ignore) {
  mIgnoreDepth = ignore;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WorldSpaceGuiArea::getIgnoreDepth() const {
  return mIgnoreDepth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WorldSpaceGuiArea::getUseLinearDepthBuffer() const {
  return mUseLinearDepthBuffer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WorldSpaceGuiArea::setUseLinearDepthBuffer(bool bEnable) {
  mShaderDirty          = true;
  mUseLinearDepthBuffer = bEnable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WorldSpaceGuiArea::calculateMousePosition(
    VistaVector3D const& vRayOrigin, VistaVector3D const& vRayEnd, int& x, int& y) {

  VistaRay   ray(vRayOrigin, vRayEnd - vRayOrigin);
  VistaPlane plane;

  VistaVector3D intersection;

  if (!plane.CalcIntersection(ray, intersection)) {
    return false;
  }

  if (intersection[0] >= -0.5f && intersection[0] <= 0.5f && intersection[1] >= -0.5f &&
      intersection[1] <= 0.5f) {
    x = (int)((intersection[0] + 0.5) * mWidth);
    y = (int)((-intersection[1] + 0.5) * mHeight);
    return true;
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WorldSpaceGuiArea::Do() {
  utils::FrameTimings::ScopedTimer timer("User Interface");
  if (mShaderDirty) {
    delete mShader;
    mShader = new VistaGLSLShader();

    std::string defines = "#version 330\n";

    if (mUseLinearDepthBuffer) {
      defines += "#define USE_LINEARDEPTHBUFFER\n";
    }

    mShader->InitVertexShaderFromString(defines + QUAD_VERT);
    mShader->InitFragmentShaderFromString(defines + QUAD_FRAG);
    mShader->Link();

    mShaderDirty = false;
  }

  if (mIgnoreDepth)
    glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
  else
    glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (mIgnoreDepth) {
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);
  }

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

  // get modelview and projection matrices
  GLfloat glMatMV[16], glMatP[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, &glMatMV[0]);
  glGetFloatv(GL_PROJECTION_MATRIX, &glMatP[0]);
  glUniformMatrix4fv(mShader->GetUniformLocation("uMatModelView"), 1, GL_FALSE, glMatMV);
  glUniformMatrix4fv(mShader->GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP);

  // draw back-to-front
  auto const& items = getItems();
  for (auto item = items.rbegin(); item != items.rend(); ++item) {
    if ((*item)->getIsEnabled()) {
      glPushMatrix();
      glTranslatef((GLfloat)((*item)->getRelPositionX() + (*item)->getRelOffsetX() - 0.5),
          (GLfloat)(-(*item)->getRelPositionY() - (*item)->getRelOffsetY() + 0.5), 0);

      glScalef((*item)->getRelSizeX(), (*item)->getRelSizeY(), 1.f);

      (*item)->getTexture()->Bind(GL_TEXTURE0);
      mShader->SetUniform(mShader->GetUniformLocation("iTexture"), 0);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      (*item)->getTexture()->Unbind(GL_TEXTURE0);

      glPopMatrix();
    }
  }

  mShader->Release();

  if (mIgnoreDepth) {
    glDepthMask(GL_TRUE);
  }

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WorldSpaceGuiArea::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float fMin[3] = {-1.f, -1.f, -0.000001f};
  float fMax[3] = {1.f, 1.f, 0.000001f};

  oBoundingBox.SetBounds(fMin, fMax);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
