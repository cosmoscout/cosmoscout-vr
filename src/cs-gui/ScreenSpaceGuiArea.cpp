////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ScreenSpaceGuiArea.hpp"

#include "../cs-utils/FrameStats.hpp"
#include "GuiItem.hpp"

#include <VistaMath/VistaBoundingBox.h>
#include <VistaMath/VistaGeometries.h>

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaOGLExt/VistaGLSLShader.h>

namespace cs::gui {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* const ScreenSpaceGuiArea::QUAD_VERT = R"(
vec2 positions[4] = vec2[](
    vec2(-0.5,  0.5),
    vec2( 0.5,  0.5),
    vec2(-0.5, -0.5),
    vec2( 0.5, -0.5)
);

uniform vec2 iPosition;
uniform vec2 iScale;

out vec2 vTexCoords;
out vec4 vPosition;

void main() {
  vec2 p = positions[gl_VertexID];
  vTexCoords = vec2(p.x, -p.y) + 0.5;
  vPosition = vec4((p*iScale + iPosition)*2-1, 0, 1);
  gl_Position = vPosition;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* const ScreenSpaceGuiArea::QUAD_FRAG = R"(
in vec2 vTexCoords;
in vec4 vPosition;

uniform sampler2D texture;
uniform ivec2 texSize;

layout(location = 0) out vec4 vOutColor;

void main() {
  vOutColor = texelFetch(texture, ivec2(texSize * vTexCoords), 0);
  if (vOutColor.a == 0.0) discard;
  vOutColor.rgb /= vOutColor.a;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

ScreenSpaceGuiArea::ScreenSpaceGuiArea(VistaViewport* pViewport)
    : mViewport(pViewport) {
  Observe(mViewport->GetViewportProperties());
  ScreenSpaceGuiArea::onViewportChange();
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
  utils::FrameStats::ScopedTimer timer("User Interface");
  if (mShaderDirty) {
    mShader = VistaGLSLShader();

    std::string defines = "#version 330\n";

    mShader.InitVertexShaderFromString(defines + QUAD_VERT);
    mShader.InitFragmentShaderFromString(defines + QUAD_FRAG);
    mShader.Link();

    mUniforms.position = mShader.GetUniformLocation("iPosition");
    mUniforms.scale    = mShader.GetUniformLocation("iScale");
    mUniforms.texSize  = mShader.GetUniformLocation("texSize");
    mUniforms.texture  = mShader.GetUniformLocation("texture");

    mShaderDirty = false;
  }

  glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(GL_FALSE);
  glDisable(GL_DEPTH_TEST);

  mShader.Bind();

  // draw back-to-front
  auto const& items = getItems();
  for (auto item = items.rbegin(); item != items.rend(); ++item) {
    auto* guiItem = *item;

    bool textureRightSize = guiItem->getWidth() == guiItem->getTextureSizeX() &&
                            guiItem->getHeight() == guiItem->getTextureSizeY();

    if (guiItem->getIsEnabled() && textureRightSize) {
      float posX = guiItem->getRelPositionX() + guiItem->getRelOffsetX();
      float posY = 1 - guiItem->getRelPositionY() - guiItem->getRelOffsetY();
      mShader.SetUniform(mUniforms.position, posX, posY);

      float scaleX = guiItem->getRelSizeX();
      float scaleY = guiItem->getRelSizeY();
      mShader.SetUniform(mUniforms.scale, scaleX, scaleY);

      glUniform2i(mUniforms.texSize, guiItem->getTextureSizeX(), guiItem->getTextureSizeY());

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, guiItem->getTexture());
      mShader.SetUniform(mUniforms.texture, 0);

      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

      glBindTexture(GL_TEXTURE_2D, 0);
    }
  }

  mShader.Release();

  glPopAttrib();

  // A viewport resize event occurred some frames ago. Let's resize our items accordingly!
  if (mDelayedViewportUpdate > 0 &&
      mDelayedViewportUpdate < GetVistaSystem()->GetGraphicsManager()->GetFrameCount()) {
    mDelayedViewportUpdate = 0;
    mViewport->GetViewportProperties()->GetSize(mWidth, mHeight);
    updateItems();
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ScreenSpaceGuiArea::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float      min(std::numeric_limits<float>::min());
  float      max(std::numeric_limits<float>::max());
  std::array fMin{min, min, min};
  std::array fMax{max, max, max};

  oBoundingBox.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ScreenSpaceGuiArea::ObserverUpdate(
    IVistaObserveable* /*pObserveable*/, int nMsg, int /*nTicket*/) {
  if (nMsg == VistaViewport::VistaViewportProperties::MSG_SIZE_CHANGE) {
    // As it's not a good idea to resize CEF gui elements very often (performance wise and sometimes
    // resize events get lost), we wait a hard-coded number of frames until we perform the actual
    // resizing.
    mDelayedViewportUpdate = GetVistaSystem()->GetGraphicsManager()->GetFrameCount() + 5;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ScreenSpaceGuiArea::onViewportChange() {
  mViewport->GetViewportProperties()->GetSize(mWidth, mHeight);
  updateItems();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
