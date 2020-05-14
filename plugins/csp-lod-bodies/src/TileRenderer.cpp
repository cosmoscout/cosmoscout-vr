////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TileRenderer.hpp"

#include "HEALPix.hpp"
#include "PlanetParameters.hpp"
#include "RenderDataDEM.hpp"
#include "RenderDataImg.hpp"
#include "TileTextureArray.hpp"
#include "TreeManagerBase.hpp"

#include "../../../src/cs-graphics/Shadows.hpp"
#include "../../../src/cs-utils/convert.hpp"

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
GLenum const texUnitNameDEM = GL_TEXTURE0;
GLint const  texUnitDEM     = 0;

GLenum const texUnitNameIMG = GL_TEXTURE1;
GLint const  texUnitIMG     = 1;

GLint const texUnitShadow = 2;

GLsizeiptr const SizeX = TileBase::SizeX; // NOLINT(cppcoreguidelines-interfaces-global-init)
GLsizeiptr const SizeY = TileBase::SizeY; // NOLINT(cppcoreguidelines-interfaces-global-init)
// number of vertices that make up a patch
GLsizeiptr const NumVertices = SizeX * SizeY;
// number of indices: (number of quads) * (2 triangles per quad)
//                                      * (3 indices per triangle)
GLsizeiptr const NumIndices = (SizeX - 1) * (SizeY - 1) * 6;

const char* BoundsVertexShaderName("VistaPlanetTileBounds.vert");
const char* BoundsFragmentShaderName("VistaPlanetTileBounds.frag");

////////////////////////////////////////////////////////////////////////////////////////////////////

// Recursively constructs the index buffer in such a way that consecutive
// (sub-)parts of the index buffer can be used to draw sub quadrants of
// the patch.

// Starts to write indices at @a buffer + @a idx and returns the offset
// for the next set of indices. The @a level, @a baseX, and @a baseY
// arguments specify which sub quadrant indices are being generated for.

// The indices for the sub quadrants are generated in the order:
// South, East, West, North. At the lowest level the quads at the south
// tip are numbered as in the following diagram:

//           \  11 / \    / \  7  /
//             \ /     \/     \ /
//       \  10 / \  9  /\  6  / \  5  /
//         \ /     \ /    \ /     \ /
//           \  8  / \  3 / \  4  /
//             \ /     \/     \ /
//               \  2  /\  1  /
//                 \ /    \ /
//                   \  0 /
//                     \/

// Numbering starts with the four quads directly at the south tip, then
// then four to the east of those, followed by the ones to the west and
// so on.
int buildTileIndices(GLuint* buffer, int idx, int level, int baseX, int baseY) {
  // number of quads along one side at this level
  int const numQuads = static_cast<int32_t>(SizeX - 1) / (int32_t(1) << level);

  if (numQuads == 1) {
    // lowest level, split single quad into triangles alternating
    // between the two patterns:
    // y1  -----        y1  -----       top row
    //     |\  |            |  /|
    //     | \ |            | / |
    //     |  \|            |/  |
    // y0  -----        y0  -----       bottom row
    //    x0   x1          x0   x1

    GLuint const x0y0 = baseY * SizeX + baseX;
    GLuint const x1y0 = baseY * SizeX + baseX + 1;
    GLuint const x0y1 = (baseY + 1) * SizeX + baseX;
    GLuint const x1y1 = (baseY + 1) * SizeX + baseX + 1;

    if ((baseX + baseY) % 2 == 0) {
      buffer[idx++] = x0y0; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      buffer[idx++] = x1y0; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      buffer[idx++] = x0y1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

      buffer[idx++] = x0y1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      buffer[idx++] = x1y0; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      buffer[idx++] = x1y1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    } else {
      buffer[idx++] = x0y1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      buffer[idx++] = x0y0; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      buffer[idx++] = x1y1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

      buffer[idx++] = x1y1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      buffer[idx++] = x0y0; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      buffer[idx++] = x1y0; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }
  } else {
    // next level "patches" are offset by half the number of quads on
    // this level
    int const nextLevel  = level + 1;
    int const baseOffset = numQuads / 2;

    idx = buildTileIndices(buffer, idx, nextLevel, baseX, baseY);
    idx = buildTileIndices(buffer, idx, nextLevel, baseX + baseOffset, baseY);
    idx = buildTileIndices(buffer, idx, nextLevel, baseX, baseY + baseOffset);
    idx = buildTileIndices(buffer, idx, nextLevel, baseX + baseOffset, baseY + baseOffset);
  }

  return idx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Calculate offset and scale factor for IMG data texture coordinates
// @a tcIMG and an offset (@a idxOffset) and size (@a idxCount) of the
// index buffer to use when rendering a tile with different resolution
// for DEM and IMG data.
void calcOffsetScale(TileId const& idDEM, TileId const& idIMG, glm::ivec3& imgOS, glm::ivec3& demOS,
    GLuint& idxCount) {
  if (idDEM.level() < idIMG.level()) {
    // image resolution is higher
    glm::int64 idx      = idIMG.patchIdx();
    int        deltaLvl = idIMG.level() - idDEM.level();

    // clamp deltaLvl to [0, 7] to avoid out of bounds access to
    // IndexOffsets{X,Y} arrays
    deltaLvl = std::min(7, deltaLvl);

    // number of indices is number of indices for full patch divided by
    // 4^(level difference) == 2^(2 * level difference)
    idxCount = NumIndices / (int64_t(1) << (2 * deltaLvl));

    imgOS = glm::ivec3(0, 0, (SizeX - 1) / (int64_t(1) << deltaLvl));
    demOS = glm::ivec3(0, 0, (SizeX - 1) / (int64_t(1) << deltaLvl));

    for (int i = deltaLvl; i > 0; --i) {
      if (idx & 0x01) {
        demOS.x += (SizeX - 1) / (int64_t(1) << i);
      }
      if (idx & 0x02) {
        demOS.y += (SizeY - 1) / (int64_t(1) << i);
      }

      idx >>= 2;
    }
  } else {
    // dtm resolution is higher or equal
    glm::int64 idx      = idDEM.patchIdx();
    int        deltaLvl = idDEM.level() - idIMG.level();

    imgOS    = glm::ivec3(0, 0, (SizeX - 1) * (int64_t(1) << deltaLvl));
    demOS    = glm::ivec3(0, 0, SizeX - 1);
    idxCount = NumIndices;

    for (int i = 0; i < deltaLvl; ++i) {
      if (idx & 0x01) {
        imgOS.x += (SizeX - 1) * (int64_t(1) << i);
      }
      if (idx & 0x02) {
        imgOS.y += (SizeY - 1) * (int64_t(1) << i);
      }

      idx >>= 2;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns the vector to upload as "VP_EdgeDelta" uniform. It stores
// the differences in resolution levels across the edges of the tile
// in order NE, NW, SW, SE.
glm::ivec4 calcEdgeDelta(RenderDataDEM* rdDEM) {
  return glm::ivec4(rdDEM->getEdgeDelta(0), rdDEM->getEdgeDelta(1), rdDEM->getEdgeDelta(2),
      rdDEM->getEdgeDelta(3));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::ivec4 calcEdgeLayerDEM(RenderDataDEM* rdDEM) {
  return glm::ivec4(rdDEM->getEdgeRData(0) ? rdDEM->getEdgeRData(0)->getTexLayer() : 0,
      rdDEM->getEdgeRData(1) ? rdDEM->getEdgeRData(1)->getTexLayer() : 0,
      rdDEM->getEdgeRData(2) ? rdDEM->getEdgeRData(2)->getTexLayer() : 0,
      rdDEM->getEdgeRData(3) ? rdDEM->getEdgeRData(3)->getTexLayer() : 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::ivec4 calcEdgeOffset(RenderDataDEM* rdDEM) {
  glm::ivec4 result(0, 0, 0, 0);

  if (rdDEM->getEdgeDelta(0) < 0) {
    RenderDataDEM* rdNE = rdDEM->getEdgeRData(0);
    assert(rdNE != nullptr);

    TileId const& idDEM = rdDEM->getTileId();
    TileId const& idNE  = rdNE->getTileId();

    glm::int64 idx      = idDEM.patchIdx();
    int        deltaLvl = idDEM.level() - idNE.level();

    for (int i = deltaLvl; i > 0; --i) {
      if (idx & 0x02) {
        result[0] += (SizeY - 1) / (int64_t(1) << i);
      }

      idx >>= 2;
    }
  }

  if (rdDEM->getEdgeDelta(1) < 0) {
    RenderDataDEM* rdNW = rdDEM->getEdgeRData(1);
    assert(rdNW != nullptr);

    TileId const& idDEM = rdDEM->getTileId();
    TileId const& idNW  = rdNW->getTileId();

    glm::int64 idx      = idDEM.patchIdx();
    int        deltaLvl = idDEM.level() - idNW.level();

    for (int i = deltaLvl; i > 0; --i) {
      if (idx & 0x01) {
        result[1] += (SizeX - 1) / (int64_t(1) << i);
      }

      idx >>= 2;
    }
  }

  if (rdDEM->getEdgeDelta(2) < 0) {
    RenderDataDEM* rdSW = rdDEM->getEdgeRData(2);
    assert(rdSW != nullptr);

    TileId const& idDEM = rdDEM->getTileId();
    TileId const& idSW  = rdSW->getTileId();

    glm::int64 idx      = idDEM.patchIdx();
    int        deltaLvl = idDEM.level() - idSW.level();

    for (int i = deltaLvl; i > 0; --i) {
      if (idx & 0x02) {
        result[2] += (SizeY - 1) / (int64_t(1) << i);
      }

      idx >>= 2;
    }
  }

  if (rdDEM->getEdgeDelta(3) < 0) {
    RenderDataDEM* rdSE = rdDEM->getEdgeRData(3);
    assert(rdSE != nullptr);

    TileId const& idDEM = rdDEM->getTileId();
    TileId const& idSE  = rdSE->getTileId();

    glm::int64 idx      = idDEM.patchIdx();
    int        deltaLvl = idDEM.level() - idSE.level();

    for (int i = deltaLvl; i > 0; --i) {
      if (idx & 0x01) {
        result[3] += (SizeX - 1) / (int64_t(1) << i);
      }

      idx >>= 2;
    }
  }

  return result;
}

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
    PlanetParameters const& params, TreeManagerBase* treeMgrDEM, TreeManagerBase* treeMgrIMG)
    : mParams(&params)
    , mTreeMgrDEM(treeMgrDEM)
    , mTreeMgrIMG(treeMgrIMG)
    , mMatVM()
    , mMatP()
    , mProgTerrain(nullptr)
    , mFrameCount(0)
    , mEnableDrawTiles(true)
    , mEnableDrawBounds(false)
    , mEnableWireframe(false)
    , mEnableFaceCulling(true) {
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
    glCullFace(GL_BACK);
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
  glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(glm::fmat4x4(mMatP)));
  loc = shader.GetUniformLocation("VP_matModelView");
  glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(glm::fmat4x4(mMatVM)));
  loc = shader.GetUniformLocation("VP_heightScale");
  shader.SetUniform(loc, static_cast<float>(mParams->mHeightScale));
  loc = shader.GetUniformLocation("VP_radius");
  shader.SetUniform(loc, static_cast<float>(mParams->mEquatorialRadius),
      static_cast<float>(mParams->mPolarRadius));
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
  locs.demAverageHeight = shader.GetUniformLocation("VP_demAverageHeight");
  locs.tileOffsetScale  = shader.GetUniformLocation("VP_tileOffsetScale");
  locs.demOffsetScale   = shader.GetUniformLocation("VP_demOffsetScale");
  locs.imgOffsetScale   = shader.GetUniformLocation("VP_imgOffsetScale");
  locs.edgeDelta        = shader.GetUniformLocation("VP_edgeDelta");
  locs.edgeLayerDEM     = shader.GetUniformLocation("VP_edgeLayerDEM");
  locs.edgeOffset       = shader.GetUniformLocation("VP_edgeOffset");
  locs.f1f2             = shader.GetUniformLocation("VP_f1f2");
  locs.layerDEM         = shader.GetUniformLocation("VP_layerDEM");
  locs.layerIMG         = shader.GetUniformLocation("VP_layerIMG");

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
    rdDEM->resetEdgeDeltas();
    rdDEM->resetEdgeRData();
    rdDEM->clearFlags();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::renderTile(RenderDataDEM* rdDEM, RenderDataImg* rdIMG, UniformLocs const& locs) {
  VistaGLSLShader& shader   = mProgTerrain->mShader;
  TileId const&    idDEM    = rdDEM->getTileId();
  GLuint           idxCount = NumIndices;

  std::array<glm::dvec2, 4> cornersLngLat{};

  glm::ivec3 demOS(0, 0, 256);
  glm::ivec3 imgOS(0, 0, 256);

  if (!rdIMG || rdDEM->getLevel() == rdIMG->getLevel()) {
    // no image data or same resolution
    cornersLngLat = HEALPix::getCornersLngLat(idDEM);
  } else if (rdDEM->getLevel() < rdIMG->getLevel()) {
    // image resolution is higher
    calcOffsetScale(idDEM, rdIMG->getTileId(), imgOS, demOS, idxCount);
    cornersLngLat = HEALPix::getCornersLngLat(rdIMG->getTileId());
  } else {
    //  elevation resolution is higher
    calcOffsetScale(idDEM, rdIMG->getTileId(), imgOS, demOS, idxCount);
    cornersLngLat = HEALPix::getCornersLngLat(idDEM);
  }

  auto  baseXY        = HEALPix::getBaseXY(idDEM);
  auto  tileOS        = glm::ivec3(baseXY.y, baseXY.z, HEALPix::getNSide(idDEM));
  auto  edgeDelta     = calcEdgeDelta(rdDEM);
  auto  edgeLayerDEM  = calcEdgeLayerDEM(rdDEM);
  auto  edgeOffset    = calcEdgeOffset(rdDEM);
  auto  patchF1F2     = glm::ivec2(HEALPix::getF1(idDEM), HEALPix::getF2(idDEM));
  float averageHeight = rdDEM->getNode()->getTile()->getMinMaxPyramid()->getAverage();

  // update uniforms
  shader.SetUniform(locs.demAverageHeight, averageHeight);
  shader.SetUniform(locs.tileOffsetScale, 3, 1, glm::value_ptr(tileOS));
  shader.SetUniform(locs.demOffsetScale, 3, 1, glm::value_ptr(demOS));
  shader.SetUniform(locs.imgOffsetScale, 3, 1, glm::value_ptr(imgOS));
  shader.SetUniform(locs.layerIMG, rdIMG ? rdIMG->getTexLayer() : 0);
  shader.SetUniform(locs.layerDEM, rdDEM->getTexLayer());
  shader.SetUniform(locs.edgeDelta, 4, 1, glm::value_ptr(edgeDelta));
  shader.SetUniform(locs.edgeLayerDEM, 4, 1, glm::value_ptr(edgeLayerDEM));
  shader.SetUniform(locs.edgeOffset, 4, 1, glm::value_ptr(edgeOffset));
  shader.SetUniform(locs.f1f2, 2, 1, glm::value_ptr(patchF1F2));

  // order of components: N, W, S, E
  std::array<glm::dvec3, 4> corners{};
  std::array<glm::dvec3, 4> normals{};
  std::array<glm::fvec3, 4> cornersViewSpace{};
  std::array<glm::fvec3, 4> normalsViewSpace{};

  glm::dmat4 matNormal = glm::transpose(glm::inverse(mMatVM));

  for (int i(0); i < 4; ++i) {
    corners.at(i) = cs::utils::convert::toCartesian(cornersLngLat.at(i), mParams->mEquatorialRadius,
        mParams->mPolarRadius, averageHeight * static_cast<float>(mParams->mHeightScale));
    cornersViewSpace.at(i) = glm::fvec3(mMatVM * glm::dvec4(corners.at(i), 1.0));

    normals.at(i) = cs::utils::convert::lngLatToNormal(
        cornersLngLat.at(i), mParams->mEquatorialRadius, mParams->mPolarRadius);
    normalsViewSpace.at(i) = glm::fvec3(matNormal * glm::dvec4(normals.at(i), 0.0));
  }

  glUniform3fv(glGetUniformLocation(shader.GetProgram(), "VP_corners"), 9,
      glm::value_ptr(cornersViewSpace[0]));
  glUniform3fv(glGetUniformLocation(shader.GetProgram(), "VP_normals"), 4,
      glm::value_ptr(normalsViewSpace[0]));

  // draw tile
  glDrawElements(GL_TRIANGLES, idxCount, GL_UNSIGNED_INT, nullptr);
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

  if (mEnableFaceCulling) {
    glDisable(GL_CULL_FACE);
    glCullFace(GL_BACK);
  }

  if (mEnableWireframe) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }

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
  glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(glm::fmat4x4(mMatP)));
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
          controlPointsViewSpace.at(i) = glm::fvec3(mMatVM * cornersWorldSpace.at(i));
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
    if (!mVboTerrain) {
      mVboTerrain = makeVBOTerrain();
    }
    if (!mIboTerrain) {
      mIboTerrain = makeIBOTerrain();
    }

    if (!mVaoTerrain) {
      mVaoTerrain = makeVAOTerrain(mVboTerrain.get(), mIboTerrain.get());
    }
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

// Construct the vertex buffer used to render a single tile.
// Contains the index in (x,y) direction of the vertex, it is converted to
// the corresponding relative position inside the patch - as well as texture
// coordinates.
std::unique_ptr<VistaBufferObject> TileRenderer::makeVBOTerrain() {
  auto             result = std::make_unique<VistaBufferObject>();
  GLsizeiptr const size   = NumVertices * sizeof(GLushort) * 2;

  result->BindAsVertexDataBuffer();
  result->BufferData(size, nullptr, GL_STATIC_DRAW);

  GLuint idx    = 0;
  auto*  buffer = static_cast<GLushort*>(result->MapBuffer(GL_WRITE_ONLY));
  for (int y = 0; y < SizeY; ++y) {
    for (int x = 0; x < SizeX; ++x) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      buffer[idx++] = static_cast<GLushort>(x);

      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      buffer[idx++] = static_cast<GLushort>(y);
    }
  }
  result->UnmapBuffer();
  result->Release();

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<VistaBufferObject> TileRenderer::makeIBOTerrain() {
  auto             result = std::make_unique<VistaBufferObject>();
  GLsizeiptr const size   = NumIndices * sizeof(GLuint);

  result->BindAsIndexBuffer();
  result->BufferData(size, nullptr, GL_STATIC_DRAW);

  int   idx    = 0;
  auto* buffer = static_cast<GLuint*>(result->MapBuffer(GL_WRITE_ONLY));

  // tile
  idx = buildTileIndices(buffer, idx, 0, 0, 0);

  result->UnmapBuffer();
  result->Release();

  assert(idx == NumIndices);

  return result;
}

// Sets up the VertexArrayObject for rendering a Tile
std::unique_ptr<VistaVertexArrayObject> TileRenderer::makeVAOTerrain(
    VistaBufferObject* vbo, VistaBufferObject* ibo) {
  auto result = std::make_unique<VistaVertexArrayObject>();
  result->Bind();
  result->EnableAttributeArray(0);
  result->SpecifyAttributeArrayInteger(0, 2, GL_UNSIGNED_SHORT, 0, 0, vbo);
  result->SpecifyIndexBufferObject(ibo, GL_UNSIGNED_INT);
  result->Release();

  return result;
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
  VistaShaderRegistry& reg = VistaShaderRegistry::GetInstance();

  auto result = std::make_unique<VistaGLSLShader>();
  result->InitVertexShaderFromString(reg.RetrieveShader(BoundsVertexShaderName));
  result->InitFragmentShaderFromString(reg.RetrieveShader(BoundsFragmentShaderName));

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

void TileRenderer::setProjection(glm::dmat4 const& m) {
  mMatP = m;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileRenderer::setModelview(glm::dmat4 const& m) {
  mMatVM = m;
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
