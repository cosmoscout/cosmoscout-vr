////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ScreenSpaceGuiArea.hpp"

#include "../cs-utils/FrameTimings.hpp"
#include "GuiItem.hpp"

#include <VistaMath/VistaBoundingBox.h>
#include <VistaMath/VistaGeometries.h>

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string QUAD_VERT = R"(
vec2 positions[4] = vec2[](
    vec2(-0.5, -0.5),
    vec2( 0.5, -0.5),
    vec2(-0.5,  0.5),
    vec2( 0.5,  0.5)
);        

uniform vec2 iPosition;                       
uniform vec2 iScale;                       

out vec2 vTexCoords;                                                            
out vec4 vPosition;
                                                                           
void main()                                                                
{                       
  vec2 p = positions[gl_VertexID];                                                   
  vTexCoords = vec2(p.x, -p.y) + 0.5;
  vPosition = vec4((p*iScale + iPosition)*2-1, 0, 1);
  gl_Position = vPosition;                                   
}                                                            
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string QUAD_FRAG = R"(
in vec2 vTexCoords;                                                             
in vec4 vPosition;

uniform sampler2D iTexture;                                                
                                                                           
layout(location = 0) out vec4 vOutColor;                                   
                                                                           
void main()                                                                
{                                                                                                    
  vOutColor = texture(iTexture, vTexCoords); 
  if (vOutColor.a == 0.0) discard; 

  vOutColor.rgb /= vOutColor.a;
}  
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

namespace cs::gui {

////////////////////////////////////////////////////////////////////////////////////////////////////

ScreenSpaceGuiArea::ScreenSpaceGuiArea(VistaViewport* pViewport)
    : mViewport(pViewport)
    , mShader(new VistaGLSLShader()) {
  Observe(mViewport->GetViewportProperties());
  onViewportChange();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ScreenSpaceGuiArea::~ScreenSpaceGuiArea() {
  delete mShader;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int ScreenSpaceGuiArea::getWidth() const {
  return mWidth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int ScreenSpaceGuiArea::getHeight() const {
  return mHeight;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ScreenSpaceGuiArea::Do() {
  utils::FrameTimings::ScopedTimer timer("User Interface");
  if (mShaderDirty) {
    delete mShader;
    mShader = new VistaGLSLShader();

    std::string defines = "#version 430 compatibility\n";

    mShader->InitVertexShaderFromString(defines + QUAD_VERT);
    mShader->InitFragmentShaderFromString(defines + QUAD_FRAG);
    mShader->Link();

    mShaderDirty = false;
  }

  glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(GL_FALSE);
  glDisable(GL_DEPTH_TEST);

  mShader->Bind();

  // draw back-to-front
  auto const& items = getItems();
  for (auto item = items.rbegin(); item != items.rend(); ++item) {
    if ((*item)->getIsEnabled()) {
      float posX = (*item)->getRelPositionX() + (*item)->getRelOffsetX();
      float posY = 1 - (*item)->getRelPositionY() - (*item)->getRelOffsetY();
      mShader->SetUniform(mShader->GetUniformLocation("iPosition"), posX, posY);

      float scaleX = (*item)->getRelSizeX();
      float scaleY = (*item)->getRelSizeY();
      mShader->SetUniform(mShader->GetUniformLocation("iScale"), scaleX, scaleY);

      (*item)->getTexture()->Bind(GL_TEXTURE0);
      mShader->SetUniform(mShader->GetUniformLocation("iTexture"), 0);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      (*item)->getTexture()->Unbind(GL_TEXTURE0);
    }
  }

  mShader->Release();

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ScreenSpaceGuiArea::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float min(std::numeric_limits<float>::min());
  float max(std::numeric_limits<float>::max());
  float fMin[3] = {min, min, min};
  float fMax[3] = {max, max, max};

  oBoundingBox.SetBounds(fMin, fMax);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ScreenSpaceGuiArea::ObserverUpdate(IVistaObserveable* pObserveable, int nMsg, int nTicket) {
  if (nMsg == VistaViewport::VistaViewportProperties::MSG_SIZE_CHANGE) {
    onViewportChange();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ScreenSpaceGuiArea::onViewportChange() {
  mViewport->GetViewportProperties()->GetSize(mWidth, mHeight);
  updateItems();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
