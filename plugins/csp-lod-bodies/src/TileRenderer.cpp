////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TileRenderer.hpp"

#include "PlanetParameters.hpp"
#include "TileNode.hpp"
#include "TileTextureArray.hpp"
#include "TreeManager.hpp"

#include "../../../src/cs-graphics/Shadows.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/filesystem.hpp"

#include <VistaBase/VistaStreamUtils.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaOGLExt/VistaShaderRegistry.h>
#include <VistaOGLExt/VistaTexture.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/io.hpp>
#include <memory>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

GLenum const texUnitNameDEM = GL_TEXTURE0;
GLint const  texUnitDEM     = 0;

GLenum const texUnitNameIMG = GL_TEXTURE1;
GLint const  texUnitIMG     = 1;

GLint const texUnitShadow = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<VistaBufferObject>      TileRenderer::mVboTerrain;
std::unique_ptr<VistaBufferObject>      TileRenderer::mIboTerrain;
std::unique_ptr<VistaVertexArrayObject> TileRenderer::mVaoTerrain;
std::unique_ptr<VistaBufferObject>      TileRenderer::mVboBounds;
std::unique_ptr<VistaBufferObject>      TileRenderer::mIboBounds;
std::unique_ptr<VistaVertexArrayObject> TileRenderer::mVaoBounds;
std::unique_ptr<VistaGLSLShader>        TileRenderer::mProgBounds;

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
TileRenderer::TileRenderer(
    PlanetParameters const& params, TreeManager* treeMgr, uint32_t tileResolution)
    : mParams(&params)
    , mTreeMgr(treeMgr)
    , mMatM()
    , mMatV()
    , mMatP()
    , mProgTerrain(nullptr)
    , mEnableDrawBounds(false)
    , mEnableWireframe(false)
    , mEnableFaceCulling(true)
    , mTileResolution(tileResolution)
    , mGridResolution(mTileResolution + 2)
    , mIndexCount((mGridResolution - 1) * (2 + 2 * mGridResolution)) {

  std::vector<uint16_t> vertices(mGridResolution * mGridResolution * 2);
  std::vector<uint32_t> indices(mIndexCount);

  for (uint32_t x = 0; x < mGridResolution; ++x) {
    for (uint32_t y = 0; y < mGridResolution; ++y) {
      vertices[(x * mGridResolution + y) * 2 + 0] = x;
      vertices[(x * mGridResolution + y) * 2 + 1] = y;
    }
  }

  uint32_t index = 0;

  for (uint32_t x = 0; x < mGridResolution - 1; ++x) {
    indices[index++] = x * mGridResolution;
    for (uint32_t y = 0; y < mGridResolution; ++y) {
      indices[index++] = x * mGridResolution + y;
      indices[index++] = (x + 1) * mGridResolution + y;
    }
    indices[index] = indices[index - 1];
    ++index;
  }

  mVaoTerrain = std::make_unique<VistaVertexArrayObject>();
  mVaoTerrain->Bind();

  mVboTerrain = std::make_unique<VistaBufferObject>();
  mVboTerrain->Bind(GL_ARRAY_BUFFER);
  mVboTerrain->BufferData(vertices.size() * sizeof(uint16_t), vertices.data(), GL_STATIC_DRAW);

  mIboTerrain = std::make_unique<VistaBufferObject>();
  mIboTerrain->Bind(GL_ELEMENT_ARRAY_BUFFER);
  mIboTerrain->BufferData(indices.size() * sizeof(uint32_t), indices.data(), GL_STATIC_DRAW);

  mVaoTerrain->EnableAttributeArray(0);
  mVaoTerrain->SpecifyAttributeArrayInteger(0, 2, GL_UNSIGNED_SHORT, 0, 0, mVboTerrain.get());

  mVaoTerrain->Release();
  mIboTerrain->Release();
  mVboTerrain->Release();

  // Now create the VBO, VAO, IBO, and shader for the bounds rendering.
  mVboBounds  = makeVBOBounds();
  mIboBounds  = makeIBOBounds();
  mVaoBounds  = makeVAOBounds(mVboBounds.get(), mIboBounds.get());
  mProgBounds = makeProgBounds();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setTerrainShader(TerrainShader* shader) {
  mProgTerrain = shader;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TerrainShader* TileRenderer::getTerrainShader() const {
  return mProgTerrain;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::render(std::vector<TileNode*> const& nodes, cs::graphics::ShadowMap* shadowMap) {

  if (!nodes.empty()) {
    preRenderTiles(shadowMap);
    renderTiles(nodes);
    postRenderTiles(shadowMap);

    if (mEnableDrawBounds) {
      preRenderBounds();
      renderBounds(nodes);
      postRenderBounds();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::preRenderTiles(cs::graphics::ShadowMap* shadowMap) {
  auto const& glDEM = mTreeMgr->getGLResources()->get(TileDataType::eElevation);
  auto const& glIMG = mTreeMgr->getGLResources()->get(TileDataType::eColor);

  // setup OpenGL state
  glPushAttrib(GL_COLOR_BUFFER_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_LIGHTING_BIT |
               GL_POLYGON_BIT | GL_TEXTURE_BIT);

  glDisable(GL_BLEND);

  if (mEnableWireframe) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  }

  if (mEnableFaceCulling) {
    glEnable(GL_CULL_FACE);
  } else {
    glDisable(GL_CULL_FACE);
  }

  // bind textures with tile data
  if (glDEM) {
    glActiveTexture(texUnitNameDEM);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, glDEM->getTextureId());
  }

  if (glIMG) {
    glActiveTexture(texUnitNameIMG);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, glIMG->getTextureId());
  }

  mVaoTerrain->Bind();
  mProgTerrain->bind();
  VistaGLSLShader& shader = mProgTerrain->mShader;

  // update "frame global" uniforms
  GLint loc = shader.GetUniformLocation("VP_matProjection");
  glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(mMatP));
  loc = shader.GetUniformLocation("VP_matModel");
  glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(glm::mat4(mMatM)));
  loc = shader.GetUniformLocation("VP_matView");
  glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(mMatV));
  loc = shader.GetUniformLocation("VP_heightScale");
  shader.SetUniform(loc, static_cast<float>(mParams->mHeightScale));
  loc = shader.GetUniformLocation("VP_radii");
  shader.SetUniform(loc, static_cast<float>(mParams->mRadii.x),
      static_cast<float>(mParams->mRadii.y), static_cast<float>(mParams->mRadii.z));
  loc = shader.GetUniformLocation("VP_texDEM");
  shader.SetUniform(loc, texUnitDEM);
  loc = shader.GetUniformLocation("VP_texIMG");
  shader.SetUniform(loc, texUnitIMG);
  loc = shader.GetUniformLocation("VP_shadowMapMode");
  shader.SetUniform(loc, shadowMap == nullptr);

  if (shadowMap) {
    shader.SetUniform(shader.GetUniformLocation("VP_shadowBias"), shadowMap->getBias());
    shader.SetUniform(shader.GetUniformLocation("VP_shadowCascades"),
        static_cast<int>(shadowMap->getMaps().size()));

    for (size_t i = 0; i < shadowMap->getMaps().size(); ++i) {
      GLint locSamplers = glGetUniformLocation(
          shader.GetProgram(), ("VP_shadowMaps[" + std::to_string(i) + "]").c_str());
      GLint locMatrices = glGetUniformLocation(shader.GetProgram(),
          ("VP_shadowProjectionViewMatrices[" + std::to_string(i) + "]").c_str());

      shadowMap->getMaps()[i]->Bind(GL_TEXTURE0 + texUnitShadow + static_cast<int>(i));
      glUniform1i(locSamplers, texUnitShadow + static_cast<int>(i));

      auto mat = shadowMap->getShadowMatrices()[i];
      glUniformMatrix4fv(locMatrices, 1, GL_FALSE, mat.GetData());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::renderTiles(std::vector<TileNode*> const& nodes) {
  VistaGLSLShader& shader = mProgTerrain->mShader;

  // query uniform locations once and store in locs
  UniformLocs locs{};
  locs.heightInfo  = shader.GetUniformLocation("VP_heightInfo");
  locs.offsetScale = shader.GetUniformLocation("VP_offsetScale");
  locs.f1f2        = shader.GetUniformLocation("VP_f1f2");
  locs.dataLayers  = shader.GetUniformLocation("VP_dataLayers");

  for (auto* node : nodes) {
    renderTile(node, locs);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::renderTile(TileNode* node, UniformLocs const& locs) {
  auto const& dem = node->getTileData(TileDataType::eElevation);
  auto const& img = node->getTileData(TileDataType::eColor);

  // Do not attempt to draw tiles with missing data.
  if (dem->getTexLayer() < 0 || (img && img->getTexLayer() < 0)) {
    return;
  }

  VistaGLSLShader& shader = mProgTerrain->mShader;

  float averageHeight = node->getMinMaxPyramid()->getAverage();
  float minHeight     = node->getMinMaxPyramid()->getMin();
  float maxHeight     = node->getMinMaxPyramid()->getMax();

  // update uniforms
  shader.SetUniform(locs.heightInfo, averageHeight, maxHeight - minHeight);
  shader.SetUniform(locs.offsetScale, 3, 1, glm::value_ptr(node->getTileOffsetScale()));
  shader.SetUniform(locs.f1f2, 2, 1, glm::value_ptr(node->getTileF1F2()));

  glUniform2i(locs.dataLayers, dem->getTexLayer(), img ? img->getTexLayer() : 0);

  // order of components: N, W, S, E
  auto const&               cornersLngLat = node->getCornersLngLat();
  std::array<glm::dvec3, 4> corners{};
  std::array<glm::dvec3, 4> normals{};
  std::array<glm::fvec3, 4> cornersWorldSpace{};
  std::array<glm::fvec3, 4> normalsWorldSpace{};

  // Convert tile corners to camera-relative coordinates in double precision.
  for (int i(0); i < 4; ++i) {
    corners.at(i)           = cs::utils::convert::toCartesian(cornersLngLat.at(i), mParams->mRadii,
        averageHeight * static_cast<float>(mParams->mHeightScale));
    cornersWorldSpace.at(i) = glm::fvec3(mMatM * glm::dvec4(corners.at(i), 1.0));

    normals.at(i)           = cs::utils::convert::lngLatToNormal(cornersLngLat.at(i));
    normalsWorldSpace.at(i) = glm::fvec3(mMatN * glm::dvec4(normals.at(i), 0.0));
  }

  glUniform3fv(glGetUniformLocation(shader.GetProgram(), "VP_corners"), 9,
      glm::value_ptr(cornersWorldSpace[0]));
  glUniform3fv(glGetUniformLocation(shader.GetProgram(), "VP_normals"), 4,
      glm::value_ptr(normalsWorldSpace[0]));

  // draw tile
  glDrawElements(GL_TRIANGLE_STRIP, mIndexCount, GL_UNSIGNED_INT, nullptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::postRenderTiles(cs::graphics::ShadowMap* shadowMap) {
  // clean up OpenGL state
  mProgTerrain->release();
  mVaoTerrain->Release();

  glActiveTexture(texUnitNameDEM);
  glBindTexture(GL_TEXTURE_2D_ARRAY, 0U);

  glActiveTexture(texUnitNameIMG);
  glBindTexture(GL_TEXTURE_2D_ARRAY, 0U);

  if (mEnableWireframe) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }

  glFrontFace(GL_CW);

  glPopAttrib();

  if (shadowMap) {
    for (auto* map : shadowMap->getMaps()) {
      map->Unbind();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::preRenderBounds() {
  // setup OpenGL state
  mVaoBounds->Bind();
  mProgBounds->Bind();

  GLint loc = mProgBounds->GetUniformLocation("VP_matProjection");
  glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(mMatP));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::renderBounds(std::vector<TileNode*> const& nodes) {
  for (auto const& it : nodes) {
    if (it->hasBounds()) {
      BoundingBox<double> const& tb = it->getBounds();

      std::array<glm::dvec4, 8> cornersWorldSpace = {
          glm::dvec4(tb.getMin().x, tb.getMin().y, tb.getMin().z, 1.0),
          glm::dvec4(tb.getMax().x, tb.getMin().y, tb.getMin().z, 1.0),
          glm::dvec4(tb.getMax().x, tb.getMin().y, tb.getMax().z, 1.0),
          glm::dvec4(tb.getMin().x, tb.getMin().y, tb.getMax().z, 1.0),

          glm::dvec4(tb.getMin().x, tb.getMax().y, tb.getMin().z, 1.0),
          glm::dvec4(tb.getMax().x, tb.getMax().y, tb.getMin().z, 1.0),
          glm::dvec4(tb.getMax().x, tb.getMax().y, tb.getMax().z, 1.0),
          glm::dvec4(tb.getMin().x, tb.getMax().y, tb.getMax().z, 1.0)};

      std::array<glm::fvec3, 8> controlPointsViewSpace{};
      for (int i(0); i < 8; ++i) {
        controlPointsViewSpace.at(i) =
            glm::fvec3(glm::dmat4(mMatV) * mMatM * cornersWorldSpace.at(i));
      }

      glUniform3fv(glGetUniformLocation(mProgBounds->GetProgram(), "VP_corners"), 8,
          glm::value_ptr(controlPointsViewSpace[0]));

      glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, nullptr);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::postRenderBounds() {
  // clean up OpenGL state
  mProgBounds->Release();
  mVaoBounds->Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<VistaBufferObject> TileRenderer::makeVBOBounds() {
  auto             result = std::make_unique<VistaBufferObject>();
  GLsizeiptr const size   = 8 * sizeof(GLubyte);

  result->BindAsVertexDataBuffer();
  result->BufferData(size, nullptr, GL_STATIC_DRAW);

  GLuint idx    = 0;
  auto*  buffer = static_cast<GLubyte*>(result->MapBuffer(GL_WRITE_ONLY));

  buffer[idx++] = 0; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 2; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 3; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 4; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 5; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 6; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 7; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

  result->UnmapBuffer();
  result->Release();

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<VistaBufferObject> TileRenderer::makeIBOBounds() {
  auto             result = std::make_unique<VistaBufferObject>();
  GLsizeiptr const size   = 24 * sizeof(GLuint);

  result->BindAsIndexBuffer();
  result->BufferData(size, nullptr, GL_STATIC_DRAW);

  GLuint idx    = 0;
  auto*  buffer = static_cast<GLuint*>(result->MapBuffer(GL_WRITE_ONLY));

  // bottom "ring"
  buffer[idx++] = 0; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 2; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 2; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 3; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 3; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 0; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

  // verticals
  buffer[idx++] = 0; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 4; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 5; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 2; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 6; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 3; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 7; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

  // top "ring"
  buffer[idx++] = 4; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 5; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 5; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 6; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 6; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 7; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 7; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  buffer[idx++] = 4; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

  result->UnmapBuffer();
  result->Release();

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Sets up the VertexArrayObject for rendering bounds of a tile.
std::unique_ptr<VistaVertexArrayObject> TileRenderer::makeVAOBounds(
    VistaBufferObject* vbo, VistaBufferObject* ibo) {
  auto result = std::make_unique<VistaVertexArrayObject>();
  result->Bind();
  result->EnableAttributeArray(0);
  result->SpecifyAttributeArrayInteger(0, 1, GL_UNSIGNED_BYTE, 0, 0, vbo);
  result->SpecifyIndexBufferObject(ibo, GL_UNSIGNED_INT);
  result->Release();

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<VistaGLSLShader> TileRenderer::makeProgBounds() {
  auto result = std::make_unique<VistaGLSLShader>();
  result->InitVertexShaderFromString(
      cs::utils::filesystem::loadToString("../share/resources/shaders/VistaPlanetTileBounds.vert"));
  result->InitFragmentShaderFromString(
      cs::utils::filesystem::loadToString("../share/resources/shaders/VistaPlanetTileBounds.frag"));
  result->Link();

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TreeManager* TileRenderer::getTreeManager() const {
  return mTreeMgr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setModel(glm::dmat4 const& m) {
  mMatM = m;
  mMatN = glm::transpose(glm::inverse(mMatM));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setView(glm::mat4 const& m) {
  mMatV = m;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setProjection(glm::mat4 const& m) {
  mMatP = m;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setDrawBounds(bool enable) {
  mEnableDrawBounds = enable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TileRenderer::getDrawBounds() const {
  return mEnableDrawBounds;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setWireframe(bool enable) {
  mEnableWireframe = enable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TileRenderer::getWireframe() const {
  return mEnableWireframe;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setFaceCulling(bool enable) {
  mEnableFaceCulling = enable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TileRenderer::getFaceCulling() const {
  return mEnableFaceCulling;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
