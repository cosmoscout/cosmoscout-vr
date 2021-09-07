////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FloorGrid.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "logger.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaDisplaySystem.h>
#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/InteractionManager/VistaUserPlatform.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <glm/gtc/type_ptr.hpp>

#include <utility>

namespace csp::vraccessibility {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* FloorGrid::VERT_SHADER = R"(
#version 330

uniform mat4  uMatModelView;
uniform mat4  uMatProjection;
uniform float uFalloff;
uniform float uOffset;
uniform float uSize;

// inputs
layout(location = 0) in vec2 iQuadPos;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;

void main() {
  vTexCoords  = vec2((iQuadPos.x + 1) / 2 * uFalloff * uSize,
                     (iQuadPos.y + 1) / 2 * uFalloff * uSize);

  vPosition   = (uMatModelView * vec4(iQuadPos.x * uFalloff, 0.0, iQuadPos.y * uFalloff, 1.0)).xyz;
  gl_Position = uMatProjection * vec4(vPosition, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* FloorGrid::FRAG_SHADER = R"(
#version 330

uniform sampler2D uTexture;
uniform float     uFarClip;
uniform float     uAlpha;
uniform vec4      uCustomColor;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

void main() {
  oColor = texture(uTexture, vTexCoords);
  if (oColor.a == 0) {
    discard;
  }
  oColor *= vec4(uCustomColor.r, uCustomColor.g, uCustomColor.b, uAlpha);
  gl_FragDepth = length(vPosition) / uFarClip;
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

FloorGrid::FloorGrid(
    std::shared_ptr<cs::core::SolarSystem> solarSystem, Plugin::Settings::Grid& gridSettings)
    : mSolarSystem(std::move(solarSystem))
    , mGridSettings(gridSettings) {

  // Create initial Quad
  std::array<glm::vec2, 4> vertices{};
  vertices[0] = glm::vec2(-1.F, -1.F);
  vertices[1] = glm::vec2(1.F, -1.F);
  vertices[2] = glm::vec2(1.F, 1.F);
  vertices[3] = glm::vec2(-1.F, 1.F);

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(vertices.size() * sizeof(glm::vec2), vertices.data(), GL_STATIC_DRAW);
  mVBO.Release();

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0, &mVBO);

  // Create shader
  mShader.InitVertexShaderFromString(VERT_SHADER);
  mShader.InitFragmentShaderFromString(FRAG_SHADER);
  mShader.Link();

  // Get Uniform Locations
  mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
  mUniforms.texture          = mShader.GetUniformLocation("uTexture");
  mUniforms.falloff          = mShader.GetUniformLocation("uFalloff");
  mUniforms.size             = mShader.GetUniformLocation("uSize");
  mUniforms.farClip          = mShader.GetUniformLocation("uFarClip");
  mUniforms.alpha            = mShader.GetUniformLocation("uAlpha");
  mUniforms.color            = mShader.GetUniformLocation("uCustomColor");

  // Load Texture
  mTexture = cs::graphics::TextureLoader::loadFromFile(gridSettings.mTexture.get());
  mTexture->SetWrapS(GL_REPEAT);
  mTexture->SetWrapR(GL_REPEAT);

  // Add to scenegraph
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  // add to GUI Node (gui-method)
  auto* platform = GetVistaSystem()
                       ->GetPlatformFor(GetVistaSystem()->GetDisplayManager()->GetDisplaySystem())
                       ->GetPlatformNode();
  mOffsetNode.reset(pSG->NewTransformNode(platform));
  mGLNode.reset(pSG->NewOpenGLNode(mOffsetNode.get(), this));

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eGui) - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FloorGrid::~FloorGrid() {
  // remove Nodes from GUI
  auto* platform = GetVistaSystem()
                       ->GetPlatformFor(GetVistaSystem()->GetDisplayManager()->GetDisplaySystem())
                       ->GetPlatformNode();
  platform->DisconnectChild(mOffsetNode.get());
  mOffsetNode->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FloorGrid::configure(Plugin::Settings::Grid& gridSettings) {
  // check if texture settings changed
  if (mGridSettings.mTexture.get() != gridSettings.mTexture.get()) {
    mTexture = cs::graphics::TextureLoader::loadFromFile(gridSettings.mTexture.get());
    mTexture->SetWrapS(GL_REPEAT);
    mTexture->SetWrapR(GL_REPEAT);
  }
  mGridSettings = gridSettings;
  // update Offset Node
  mOffsetNode->SetTranslation(0.0F, mGridSettings.mOffset.get(), 0.0F);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FloorGrid::update() {
  mOffsetNode->SetTranslation(0.0F, mGridSettings.mOffset.get(), 0.0F);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FloorGrid::Do() {
  // do nothing if grid is disabled
  if (!mGridSettings.mEnabled.get()) {
    return true;
  }

  cs::utils::FrameTimings::ScopedTimer timer("VRAccessibility-FloorGrid");

  mShader.Bind();

  // Get modelview and projection matrices
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  // Set uniforms
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glMatMV.data());
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
  mShader.SetUniform(mUniforms.texture, 0);
  mShader.SetUniform(mUniforms.falloff, mGridSettings.mFalloff.get());
  mShader.SetUniform(mUniforms.size, mGridSettings.mSize.get());
  mShader.SetUniform(mUniforms.farClip, cs::utils::getCurrentFarClipDistance());
  mShader.SetUniform(mUniforms.alpha, mGridSettings.mAlpha.get());
  glUniform4fv(mUniforms.color, 1,
      glm::value_ptr(Plugin::GetColorFromHexString(mGridSettings.mColor.get())));

  // Bind Texture
  mTexture->Bind(GL_TEXTURE0);

  // Draw
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  mVAO.Bind();
  glDrawArrays(GL_QUADS, 0, 4);
  mVAO.Release();

  // Clean Up
  mTexture->Unbind(GL_TEXTURE0);
  glDisable(GL_BLEND);

  mShader.Release();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FloorGrid::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::vraccessibility