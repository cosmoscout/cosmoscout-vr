////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TileRenderer.hpp"

#include "HEALPix.hpp"
#include "PlanetParameters.hpp"
#include "RenderDataDEM.hpp"
#include "RenderDataImg.hpp"
#include "TileTextureArray.hpp"
#include "TreeManagerBase.hpp"

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
TileRenderer::TileRenderer(PlanetParameters const& params, uint32_t tileResolution)
    : mParams(&params)
    , mTreeMgrDEM(nullptr)
    , mTreeMgrIMG(nullptr)
    , mMatM()
    , mMatV()
    , mMatP()
    , mProgTerrain(nullptr)
    , mFrameCount(0)
    , mEnableDrawTiles(true)
    , mEnableDrawBounds(false)
    , mEnableWireframe(false)
    , mEnableFaceCulling(true)
    , mTileResolution(tileResolution) {
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

void TileRenderer::render(std::vector<RenderData*> const& reqDEM,
    std::vector<RenderData*> const& reqIMG, cs::graphics::ShadowMap* shadowMap) {
  init();

  if (mEnableDrawTiles && !reqDEM.empty()) {
    preRenderTiles(shadowMap);
    renderTiles(reqDEM, reqIMG);
    postRenderTiles(shadowMap);
  }

  if (mEnableDrawBounds && !reqDEM.empty()) {
    preRenderBounds();
    renderBounds(reqDEM, reqIMG);
    postRenderBounds();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::preRenderTiles(cs::graphics::ShadowMap* shadowMap) {
  TileTextureArray* glDEM = mTreeMgrDEM ? &mTreeMgrDEM->getTileTextureArray() : nullptr;
  TileTextureArray* glIMG = mTreeMgrIMG ? &mTreeMgrIMG->getTileTextureArray() : nullptr;

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
    glBindTexture(GL_TEXTURE_2D_ARRAY, glDEM->getTextureId());
  }

  if (glIMG) {
    glActiveTexture(texUnitNameIMG);
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

void TileRenderer::renderTiles(
    std::vector<RenderData*> const& renderDEM, std::vector<RenderData*> const& renderIMG) {
  VistaGLSLShader& shader = mProgTerrain->mShader;

  // query uniform locations once and store in locs
  UniformLocs locs{};
  locs.heightInfo  = shader.GetUniformLocation("VP_heightInfo");
  locs.offsetScale = shader.GetUniformLocation("VP_offsetScale");
  locs.f1f2        = shader.GetUniformLocation("VP_f1f2");
  locs.dataLayers  = shader.GetUniformLocation("VP_dataLayers");

  int missingDEM = 0;
  int missingIMG = 0;

  // iterate over both std::vector<RenderData*>s together
  for (size_t i(0); i < renderDEM.size(); ++i) {
    // get data associated with nodes
    auto*          rdDEM = dynamic_cast<RenderDataDEM*>(renderDEM[i]);
    RenderDataImg* rdIMG =
        i < renderIMG.size() ? dynamic_cast<RenderDataImg*>(renderIMG[i]) : nullptr;

    // count cases of data not being on GPU ...
    if (rdDEM->getTexLayer() < 0) {
      ++missingDEM;
    }

    if (rdIMG && rdIMG->getTexLayer() < 0) {
      ++missingIMG;
    }

    // ... but do not attempt to draw
    if (rdDEM->getTexLayer() < 0 || (rdIMG && rdIMG->getTexLayer() < 0)) {
      continue;
    }

    // render
    renderTile(rdDEM, rdIMG, locs);
  }

  if (missingDEM || missingIMG) {
    // The only time this is "expected" to happen is after a texture had
    // to be resized - otherwise it suggests a bug in the resource
    // handling
    vstr::warnp() << "[TileRenderer::renderTiles]" << std::endl;
    vstr::warnp() << "Some tiles were not available on the GPU (" << missingDEM << " / "
                  << missingIMG << "  DEM/IMG)." << std::endl;
  }

  // Iterate over std::vector<RenderData*>s a second time, reset edge deltas and flags.
  // Cannot be done during rendering because a TileNode/RenderData may
  // appear multiple times in renderDEM/renderIMG.
  for (auto* it : renderDEM) {
    auto* rdDEM = dynamic_cast<RenderDataDEM*>(it);
    rdDEM->clearFlags();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::renderTile(RenderDataDEM* rdDEM, RenderDataImg* rdIMG, UniformLocs const& locs) {
  VistaGLSLShader& shader = mProgTerrain->mShader;
  TileId const&    idDEM  = rdDEM->getTileId();

  uint32_t gridResolution = mTileResolution + 2;
  uint32_t idxCount       = (gridResolution - 1) * (2 + 2 * gridResolution);

  std::array<glm::dvec2, 4> cornersLngLat{};

  cornersLngLat = HEALPix::getCornersLngLat(idDEM);

  auto  baseXY        = HEALPix::getBaseXY(idDEM);
  auto  tileOS        = glm::ivec3(baseXY.y, baseXY.z, HEALPix::getNSide(idDEM));
  auto  patchF1F2     = glm::ivec2(HEALPix::getF1(idDEM), HEALPix::getF2(idDEM));
  float averageHeight = rdDEM->getNode()->getTile()->getMinMaxPyramid()->getAverage();
  float minHeight     = rdDEM->getNode()->getTile()->getMinMaxPyramid()->getMin();
  float maxHeight     = rdDEM->getNode()->getTile()->getMinMaxPyramid()->getMax();

  // update uniforms
  shader.SetUniform(locs.heightInfo, averageHeight, maxHeight - minHeight);
  shader.SetUniform(locs.offsetScale, 3, 1, glm::value_ptr(tileOS));
  shader.SetUniform(locs.f1f2, 2, 1, glm::value_ptr(patchF1F2));
  glUniform2i(locs.dataLayers, rdDEM->getTexLayer(), rdIMG ? rdIMG->getTexLayer() : 0);

  // order of components: N, W, S, E
  std::array<glm::dvec3, 4> corners{};
  std::array<glm::dvec3, 4> normals{};
  std::array<glm::fvec3, 4> cornersWorldSpace{};
  std::array<glm::fvec3, 4> normalsWorldSpace{};

  glm::dmat4 matNormal = glm::transpose(glm::inverse(mMatM));

  for (int i(0); i < 4; ++i) {
    corners.at(i)           = cs::utils::convert::toCartesian(cornersLngLat.at(i), mParams->mRadii,
        averageHeight * static_cast<float>(mParams->mHeightScale));
    cornersWorldSpace.at(i) = glm::fvec3(mMatM * glm::dvec4(corners.at(i), 1.0));

    normals.at(i)           = cs::utils::convert::lngLatToNormal(cornersLngLat.at(i));
    normalsWorldSpace.at(i) = glm::fvec3(matNormal * glm::dvec4(normals.at(i), 0.0));
  }

  glUniform3fv(glGetUniformLocation(shader.GetProgram(), "VP_corners"), 9,
      glm::value_ptr(cornersWorldSpace[0]));
  glUniform3fv(glGetUniformLocation(shader.GetProgram(), "VP_normals"), 4,
      glm::value_ptr(normalsWorldSpace[0]));

  // draw tile
  glDrawElements(GL_TRIANGLE_STRIP, idxCount, GL_UNSIGNED_INT, nullptr);
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

void TileRenderer::renderBounds(
    std::vector<RenderData*> const& reqDEM, std::vector<RenderData*> const& reqIMG) {
  auto renderBounds = [this](std::vector<RenderData*> const& req) {
    for (auto const& it : req) {
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
  };

  renderBounds(reqDEM);
  renderBounds(reqIMG);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::postRenderBounds() {
  // clean up OpenGL state
  mProgBounds->Release();
  mVaoBounds->Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::init() const {
  if (mEnableDrawTiles) {

    // The vertex buffer contains a grid of vertices with an additional vertex in all directions for
    // the skirt around each patch.
    uint32_t gridResolution = mTileResolution + 2;

    std::vector<uint16_t> vertices(gridResolution * gridResolution * 2);
    std::vector<uint32_t> indices((gridResolution - 1) * (2 + 2 * gridResolution));

    for (uint32_t x = 0; x < gridResolution; ++x) {
      for (uint32_t y = 0; y < gridResolution; ++y) {
        vertices[(x * gridResolution + y) * 2 + 0] = x;
        vertices[(x * gridResolution + y) * 2 + 1] = y;
      }
    }

    uint32_t index = 0;

    for (uint32_t x = 0; x < gridResolution - 1; ++x) {
      indices[index++] = x * gridResolution;
      for (uint32_t y = 0; y < gridResolution; ++y) {
        indices[index++] = x * gridResolution + y;
        indices[index++] = (x + 1) * gridResolution + y;
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
  }

  if (mEnableDrawBounds) {
    if (!mVboBounds) {
      mVboBounds = makeVBOBounds();
    }

    if (!mIboBounds) {
      mIboBounds = makeIBOBounds();
    }

    if (!mVaoBounds) {
      mVaoBounds = makeVAOBounds(mVboBounds.get(), mIboBounds.get());
    }
    if (!mProgBounds) {
      mProgBounds = makeProgBounds();
    }
  }
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

// Sets up the VertexArrayObject for rendering bounds of a Tile
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

TreeManagerBase* TileRenderer::getTreeManagerDEM() const {
  return mTreeMgrDEM;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setTreeManagerDEM(TreeManagerBase* treeMgr) {
  mTreeMgrDEM = treeMgr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TreeManagerBase* TileRenderer::getTreeManagerIMG() const {
  return mTreeMgrIMG;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setTreeManagerIMG(TreeManagerBase* treeMgr) {
  mTreeMgrIMG = treeMgr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setFrameCount(int frameCount) {
  mFrameCount = frameCount;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setModel(glm::dmat4 const& m) {
  mMatM = m;
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

void TileRenderer::setDrawTiles(bool enable) {
  mEnableDrawTiles = enable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TileRenderer::getDrawTiles() const {
  return mEnableDrawTiles;
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
