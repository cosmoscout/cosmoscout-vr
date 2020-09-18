////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FloorGrid.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <utility>

namespace csp::floorgrid {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* FloorGrid::VERT_SHADER = R"(
#version 330

uniform mat4 uMatModelView;
uniform mat4 uMatProjection;
uniform float uFalloff;
uniform float uOffset;
uniform float uSize;

// inputs
layout(location = 0) in vec2 iQuadPos;

// outputs
out vec2 vTexCoords
out vec3 vPosition;

void main()
{
  vTexCoords  = vec2( ((iQuadPos.x + 1) / 2) * uFalloff * uSize,
                      ((iQuadPos.y + 1) / 2) * uFalloff * uSize );

  vPosition   = (uMatModelView * vec4(iQuadPos.x * uFalloff, uOffset, iQuadPos.y * uFalloff, 1.0)).xyz;
  gl_Position = uMatProjection * vec4(vPosition, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* FloorGrid::FRAG_SHADER = R"(
#version 330

uniform sampler2D uTexture;
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

FloorGrid::FloorGrid(std::shared_ptr<cs::core::SolarSystem> solarSystem)
    : mSolarSystem(std::move(solarSystem)) {

  std::vector<glm::vec2> vertices(4);
  vertices[0] = glm::vec2(-1.F, -1.F);
  vertices[1] = glm::vec2(1.F, -1.F);
  vertices[2] = glm::vec2(1.F, 1.F);
  vertices[3] = glm::vec2(-1.F, 1.F);



  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(vertices.size() * sizeof(glm::vec2), vertices.data(), GL_STATIC_DRAW);
  mVBO.Release();

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(
      0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0, &mVBO
      );

  // Create shader
  mShader.InitVertexShaderFromString(VERT_SHADER);
  mShader.InitFragmentShaderFromString(FRAG_SHADER);
  mShader.Link();

  // Add to scenegraph rings-method
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eGui) - 1
      );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FloorGrid::~FloorGrid() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FloorGrid::configure(const Plugin::Settings& settings) {
  if (mGridSettings.mTexture.get() != settings.mTexture.get()) {
    mTexture = cs::graphics::TextureLoader::loadFromFile(settings.mTexture.get());
  }
  mGridSettings = settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FloorGrid::Do() {
  if (!mGridSettings.mEnabled.get()){
    return true;
  }

  cs::utils::FrameTimings::ScopedTimer timer("FloorGrid");

  mShader.Bind();

  // Get modelview and projection matrices
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  // Set uniforms
  glUniformMatrix2fv(
      mShader.GetUniformLocation("uMatModelView"), 1, GL_FALSE, glMatP.data()
      );
  glUniformMatrix2fv(
      mShader.GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP.data()
      );
  mShader.SetUniform(
      mShader.GetUniformLocation("uTexture"), 0
      );
  mShader.SetUniform(
      mShader.GetUniformLocation("uFalloff"), mGridSettings.mFalloff.get()
      );
  mShader.SetUniform(
      mShader.GetUniformLocation("uOffset"), mGridSettings.mOffset.get()
      );
  mShader.SetUniform(
      mShader.GetUniformLocation("uSize"), mGridSettings.mSize.get()
      );

  // Bind Texture
  mTexture->Bind(GL_TEXTURE0);
  glTexParameteri(GL_TEXTURE0, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE0, GL_TEXTURE_WRAP_T, GL_REPEAT);

  // Draw
  mVAO.Bind();
  glDrawArrays(GL_QUAD_STRIP, 0, 4);
  mVAO.Release();

  // Clean Up
  mTexture->Unbind(GL_TEXTURE0);

  mShader.Release();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FloorGrid::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::floorgrid
