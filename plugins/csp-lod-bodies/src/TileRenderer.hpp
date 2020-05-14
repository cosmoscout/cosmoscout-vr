////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TILERENDERER_HPP
#define CSP_LOD_BODIES_TILERENDERER_HPP

#include "TerrainShader.hpp"
#include "TileId.hpp"

#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>
#include <boost/noncopyable.hpp>
#include <vector>

namespace cs::graphics {
class ShadowMap;
}

namespace csp::lodbodies {

struct PlanetParameters;
class TileNode;
class RenderData;
class RenderDataDEM;
class RenderDataImg;
class TreeManagerBase;

/// Renders tiles with elevation (DEM) and optionally image (IMG) data.
class TileRenderer : private boost::noncopyable {
 public:
  explicit TileRenderer(PlanetParameters const& params, TreeManagerBase* treeMgrDEM = nullptr,
      TreeManagerBase* treeMgrIMG = nullptr);

  TreeManagerBase* getTreeManagerDEM() const;
  void             setTreeManagerDEM(TreeManagerBase* treeMgr);

  TreeManagerBase* getTreeManagerIMG() const;
  void             setTreeManagerIMG(TreeManagerBase* treeMgr);

  /// Set the shader for rendering terrain tiles. Initially (or when shader is nullptr) a
  /// default shader is used. The shader must declare certain inputs and uniforms detailed below.
  ///
  /// @code
  /// | Kind    | Type           | Name                 | Description |
  /// |---------|----------------|----------------------|-------------|
  /// | uniform | vec3           | VP_PatchOffsetScale  |             |
  /// | uniform | vec3           | VP_IMG_TCOffsetScale |             |
  /// | uniform | ivec4          | VP_EdgeDelta         |             |
  /// | uniform | ivec2          | VP_f1f2              |             |
  /// | uniform | int            | VP_LayerDEM          |             |
  /// | uniform | int            | VP_LayerIMG          |             |
  /// | uniform | vec3           | VP_Radius            |             |
  /// | uniform | float          | VP_HeightScale       |             |
  /// | uniform | sampler2DArray | VP_TexDEM            |             |
  /// | uniform | sampler2DArray | VP_TexIMG            |             |
  /// | in      | ivec2          | vtxPosition          |             |
  /// @endcode
  void setTerrainShader(TerrainShader* shader);

  /// Returns the currently set shader for rendering terrain tiles.
  TerrainShader* getTerrainShader() const;

  void setFrameCount(int frameCount);
  void setProjection(glm::dmat4 const& m);
  void setModelview(glm::dmat4 const& m);

  /// Render the elevation and image tiles in reqDEM and reqIMG respectively.
  void render(std::vector<RenderData*> const& reqDEM, std::vector<RenderData*> const& reqIMG,
      cs::graphics::ShadowMap* shadowMap);

  /// Enable or disable drawing of tiles.
  void setDrawTiles(bool enable);
  bool getDrawTiles() const;

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
    GLint demAverageHeight;
    GLint tileOffsetScale;
    GLint demOffsetScale;
    GLint imgOffsetScale;
    GLint edgeDelta;
    GLint edgeLayerDEM;
    GLint edgeOffset;
    GLint f1f2;
    GLint layerDEM;
    GLint layerIMG;
  };

  void preRenderTiles(cs::graphics::ShadowMap* shadowMap);
  void renderTiles(
      std::vector<RenderData*> const& renderDEM, std::vector<RenderData*> const& renderIMG);
  void renderTile(RenderDataDEM* rdDEM, RenderDataImg* rdIMG, UniformLocs const& locs);
  void postRenderTiles(cs::graphics::ShadowMap* shadowMap);

  void preRenderBounds();
  void renderBounds(std::vector<RenderData*> const& reqDEM, std::vector<RenderData*> const& reqIMG);
  static void postRenderBounds();

  void                                           init() const;
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
  TreeManagerBase*        mTreeMgrDEM;
  TreeManagerBase*        mTreeMgrIMG;

  glm::dmat4 mMatVM;
  glm::dmat4 mMatP;

  static std::unique_ptr<VistaBufferObject>      mVboTerrain;
  static std::unique_ptr<VistaBufferObject>      mIboTerrain;
  static std::unique_ptr<VistaVertexArrayObject> mVaoTerrain;
  TerrainShader*                                 mProgTerrain;

  static std::unique_ptr<VistaBufferObject>      mVboBounds;
  static std::unique_ptr<VistaBufferObject>      mIboBounds;
  static std::unique_ptr<VistaVertexArrayObject> mVaoBounds;
  static std::unique_ptr<VistaGLSLShader>        mProgBounds;

  int  mFrameCount;
  bool mEnableDrawTiles;
  bool mEnableDrawBounds;
  bool mEnableWireframe;
  bool mEnableFaceCulling;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILERENDERER_HPP
