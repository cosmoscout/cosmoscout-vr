////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WorldSpaceGuiArea.hpp"

#include "../cs-utils/FrameStats.hpp"
#include "GuiItem.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaMath/VistaGeometries.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace cs::gui {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* const WorldSpaceGuiArea::QUAD_VERT = R"(
vec2 positions[4] = vec2[](
    vec2(-0.5, 0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, -0.5),
    vec2(0.5, -0.5)
);

uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

out vec2 vTexCoords;

void main()
{
  vec2 p = positions[gl_VertexID];
  vTexCoords = vec2(p.x, -p.y) + 0.5;
  vec4 pos = uMatModelView * vec4(p, 0, 1);
  gl_Position = uMatProjection * pos;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* const WorldSpaceGuiArea::QUAD_FRAG = R"(
in vec2 vTexCoords;

uniform sampler2D texture;

layout(location = 0) out vec4 vOutColor;

vec4 getTexel(ivec2 p) {
  return texelFetch(texture, p, 0).bgra;
}

void main() {
  vOutColor = texture2D(texture, vTexCoords);
  if (vOutColor.a == 0.0) discard;

  vOutColor.rgb /= vOutColor.a;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

WorldSpaceGuiArea::WorldSpaceGuiArea(int width, int height)
    : mWidth(width)
    , mHeight(height) {
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

void WorldSpaceGuiArea::setEnableBackfaceCulling(bool enable) {
  mBackfaceCulling = enable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WorldSpaceGuiArea::getEnableBackfaceCulling() const {
  return mBackfaceCulling;
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

  x = static_cast<int>((intersection[0] + 0.5) * mWidth);
  y = static_cast<int>((-intersection[1] + 0.5) * mHeight);

  return intersection[0] >= -0.5F && intersection[0] <= 0.5F && intersection[1] >= -0.5F &&
         intersection[1] <= 0.5F;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WorldSpaceGuiArea::Do() {
  utils::FrameStats::ScopedTimer timer("User Interface");
  if (mShaderDirty) {
    mShader = VistaGLSLShader();

    std::string defines = "#version 330\n";

    mShader.InitVertexShaderFromString(defines + QUAD_VERT);
    mShader.InitFragmentShaderFromString(defines + QUAD_FRAG);
    mShader.Link();

    mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
    mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
    mUniforms.texture          = mShader.GetUniformLocation("texture");

    mShaderDirty = false;
  }

  if (mIgnoreDepth) {
    glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
  } else {
    glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);
  }

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (mIgnoreDepth) {
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);
  }

  if (!mBackfaceCulling) {
    glDisable(GL_CULL_FACE);
  }

  mShader.Bind();

  // get modelview and projection matrices
  std::array<GLfloat, 16> glMat{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMat.data());
  glm::mat4 modelViewMat = glm::make_mat4(glMat.data());

  glGetFloatv(GL_PROJECTION_MATRIX, glMat.data());
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMat.data());

  // draw back-to-front
  auto const& items = getItems();
  for (auto item = items.rbegin(); item != items.rend(); ++item) {
    auto* guiItem = *item;

    bool textureRightSize = guiItem->getWidth() == guiItem->getTextureSizeX() &&
                            guiItem->getHeight() == guiItem->getTextureSizeY();

    if (guiItem->getIsEnabled() && textureRightSize) {
      auto localMat = glm::translate(
          modelViewMat, glm::vec3(guiItem->getRelPositionX() + guiItem->getRelOffsetX() - 0.5,
                            -guiItem->getRelPositionY() - guiItem->getRelOffsetY() + 0.5, 0.0));
      localMat =
          glm::scale(localMat, glm::vec3(guiItem->getRelSizeX(), guiItem->getRelSizeY(), 1.F));

      glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(localMat));

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, guiItem->getTexture());
      mShader.SetUniform(mUniforms.texture, 0);

      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

      glBindTexture(GL_TEXTURE_2D, 0);
    }
  }

  mShader.Release();

  if (mIgnoreDepth) {
    glDepthMask(GL_TRUE);
  }

  if (!mBackfaceCulling) {
    glEnable(GL_CULL_FACE);
  }

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WorldSpaceGuiArea::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float const          epsilon = 0.000001F;
  std::array<float, 3> fMin    = {-1.F, -1.F, -epsilon};
  std::array<float, 3> fMax    = {1.F, 1.F, epsilon};

  oBoundingBox.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
