////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILERENDERER_HPP
#define CSP_LOD_BODIES_TILERENDERER_HPP

#include "TerrainShader.hpp"
#include "TileId.hpp"

#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>
#include <vector>

namespace cs::graphics {
class ShadowMap;
}

namespace csp::lodbodies {

struct PlanetParameters;
class TileNode;
class RenderData;
class TreeManager;

/// Renders tiles with elevation (DEM) and optionally image (IMG) data.
class TileRenderer {
 public:
  explicit TileRenderer(PlanetParameters const& params, uint32_t tileResolution);
  virtual ~TileRenderer() = default;

  TileRenderer(TileRenderer const& other) = delete;
  TileRenderer(TileRenderer&& other)      = delete;

  TileRenderer& operator=(TileRenderer const& other) = delete;
  TileRenderer& operator=(TileRenderer&& other) = delete;

  TreeManager* getTreeManagerDEM() const;
  void         setTreeManagerDEM(TreeManager* treeMgr);

  TreeManager* getTreeManagerIMG() const;
  void         setTreeManagerIMG(TreeManager* treeMgr);

  /// Set the shader for rendering terrain tiles. Initially (or when shader is nullptr) a
  /// default shader is used. The shader must declare certain inputs and uniforms detailed below.
  void setTerrainShader(TerrainShader* shader);

  /// Returns the currently set shader for rendering terrain tiles.
  TerrainShader* getTerrainShader() const;

  void setFrameCount(int frameCount);
  void setModel(glm::dmat4 const& m);
  void setView(glm::mat4 const& m);
  void setProjection(glm::mat4 const& m);

  /// Render the elevation and image tiles in reqDEM and reqIMG respectively.
  void render(std::vector<RenderData*> const& reqDEM, std::vector<RenderData*> const& reqIMG,
      cs::graphics::ShadowMap* shadowMap);

  /// Enable or disable drawing of tile bounding boxes.
  void setDrawBounds(bool enable);
  bool getDrawBounds() const;

  /// Enable or disable wireframe rendering.
  void setWireframe(bool enable);
  bool getWireframe() const;

  /// Enable or disable OpenGL backface culling.
  void setFaceCulling(bool enable);
  bool getFaceCulling() const;

 private:
  struct UniformLocs {
    GLint heightInfo;
    GLint offsetScale;
    GLint f1f2;
    GLint dataLayers;
  };

  void preRenderTiles(cs::graphics::ShadowMap* shadowMap);
  void renderTiles(
      std::vector<RenderData*> const& renderDEM, std::vector<RenderData*> const& renderIMG);
  void renderTile(RenderData* rdDEM, RenderData* rdIMG, UniformLocs const& locs);
  void postRenderTiles(cs::graphics::ShadowMap* shadowMap);

  void preRenderBounds();
  void renderBounds(std::vector<RenderData*> const& reqDEM, std::vector<RenderData*> const& reqIMG);
  static void postRenderBounds();

  static std::unique_ptr<VistaBufferObject>      makeVBOTerrain();
  static std::unique_ptr<VistaBufferObject>      makeIBOTerrain();
  static std::unique_ptr<VistaVertexArrayObject> makeVAOTerrain(
      VistaBufferObject* vbo, VistaBufferObject* ibo);

  static std::unique_ptr<VistaBufferObject>      makeVBOBounds();
  static std::unique_ptr<VistaBufferObject>      makeIBOBounds();
  static std::unique_ptr<VistaVertexArrayObject> makeVAOBounds(
      VistaBufferObject* vbo, VistaBufferObject* ibo);
  static std::unique_ptr<VistaGLSLShader> makeProgBounds();

  PlanetParameters const* mParams;
  TreeManager*            mTreeMgrDEM;
  TreeManager*            mTreeMgrIMG;

  glm::dmat4 mMatM;
  glm::mat4  mMatV;
  glm::mat4  mMatP;

  static std::unique_ptr<VistaBufferObject>      mVboTerrain;
  static std::unique_ptr<VistaBufferObject>      mIboTerrain;
  static std::unique_ptr<VistaVertexArrayObject> mVaoTerrain;
  TerrainShader*                                 mProgTerrain;

  static std::unique_ptr<VistaBufferObject>      mVboBounds;
  static std::unique_ptr<VistaBufferObject>      mIboBounds;
  static std::unique_ptr<VistaVertexArrayObject> mVaoBounds;
  static std::unique_ptr<VistaGLSLShader>        mProgBounds;

  int  mFrameCount;
  bool mEnableDrawBounds;
  bool mEnableWireframe;
  bool mEnableFaceCulling;

  // The mTileResolution describes the number of vertices which are used in x and y direction for
  // rendering the elevation data. The mGridResolution is the actual amount of vertices in x and y
  // direction which is drawn, which also includes the additional vertices required for the skirt
  // around the tile. mIndexCount contains the number of entries required in the index buffer to
  // draw this grid.
  const uint32_t mTileResolution;
  const uint32_t mGridResolution;
  const uint32_t mIndexCount;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILERENDERER_HPP
