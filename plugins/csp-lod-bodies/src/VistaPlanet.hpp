////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_VISTAPLANET_HPP
#define CSP_LOD_BODIES_VISTAPLANET_HPP

#include "LODVisitor.hpp"
#include "PlanetParameters.hpp"
#include "TileRenderer.hpp"
#include "TreeManager.hpp"

#include "../../../src/cs-graphics/Shadows.hpp"
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

class VistaGLSLShader;
class VistaSystem;

namespace csp::lodbodies {

class TileBase;
class TileNode;
class TileSource;
class RenderDataDEM;
class RenderDataImg;
class TerrainShader;

/// Renders a planet from databases of hierarchical tiles.
///
/// This implements the IVistaOpenGLDraw interface and as such is added to a ViSTA scene with a
/// VistaOpenGLNode, for example:
///
/// @code{.cpp}
/// VistaSystem* vsys   = new VistaSystem();
/// // configure and init VistaSystem
///
/// VistaSceneGraph* vsg     = vsys->GetGraphicsManager()->GetSceneGraph();
/// VistaPlanet*     planet  = new VistaPlanet(vsys);
/// VistaOpenGLNode* planetN = vsg->NewOpenGLNode(vsg->GetRoot(), planet);
/// @endcode
///
/// Additionally, VistaPlanet provides the interface to set the data sources that are used to obtain
/// data (setDEMSource, setIMGSource).
class VistaPlanet : public IVistaOpenGLDraw, public cs::graphics::ShadowCaster {
 public:
  explicit VistaPlanet(std::shared_ptr<GLResources> const& glResources);

  VistaPlanet(VistaPlanet const& other) = delete;
  VistaPlanet(VistaPlanet&& other)      = delete;

  VistaPlanet& operator=(VistaPlanet const& other) = delete;
  VistaPlanet& operator=(VistaPlanet&& other) = delete;

  ~VistaPlanet() override;

  void doShadows() override;
  bool getWorldTransform(VistaTransformMatrix& matTransform) const override;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

  void       setWorldTransform(glm::dmat4 const& mat);
  glm::dmat4 getWorldTransform() const;

  /// Sets shader to use for terrain rendering. This class does not take ownership of the passed in
  /// object.
  void setTerrainShader(TerrainShader* shader);

  /// Returns the currently active shader for terrain rendering.
  TerrainShader* getTerrainShader() const;

  /// Sets the tile source for elevation data. This class does not take ownership of the passed in
  /// object.
  void setDEMSource(TileSource* srcDEM);

  /// Returns the currently active source for elevation data.
  TileSource* getDEMSource() const;

  /// Set the tile source for image data. This class does not take ownership of the passed in
  /// object.
  void setIMGSource(TileSource* srcIMG);

  /// Returns the currently active source for image data.
  TileSource* getIMGSource() const;

  /// Set planet equatorial radius. This is a potentially expensive operation since it invalidates
  /// the cached bounding volume for all tiles and requires recalculating them.
  void   setEquatorialRadius(float radius);
  double getEquatorialRadius() const;

  /// Set planet polar radius. This is a potentially expensive operation since it invalidates
  /// the cached bounding volume for all tiles and requires recalculating them.
  void   setPolarRadius(float radius);
  double getPolarRadius() const;

  /// Set factor by which to scale height data. This is a potentially expensive operation since it
  /// invalidates the cached bounding volume for all tiles and requires recalculating them.
  void   setHeightScale(float scale);
  double getHeightScale() const;

  void   setLODFactor(float lodFactor);
  double getLODFactor() const;

  void setMinLevel(int minLevel);
  int  getMinLevel() const;

  /// Returns the TileRenderer instance used to render this VistaPlanet.
  TileRenderer&       getTileRenderer();
  TileRenderer const& getTileRenderer() const;

  /// Returns the LODVisitor instance used to determine which tiles to render and load.
  LODVisitor&       getLODVisitor();
  LODVisitor const& getLODVisitor() const;

 private:
  void doFrame();
  void updateStatistics(int frameCount);
  void updateTileBounds();
  void updateTileTrees(int frameCount);
  void traverseTileTrees(int frameCount, glm::dmat4 const& matVM, glm::fmat4x4 const& matP,
      glm::ivec4 const& viewport);
  void processLoadRequests();
  void renderTiles(int frameCount, glm::dmat4 const& matVM, glm::fmat4x4 const& matP,
      cs::graphics::ShadowMap* shadowMap);

  glm::dmat4        getModelviewMatrix() const;
  static glm::dmat4 getProjectionMatrix();
  static glm::ivec4 getViewport();

  static glm::uint8 const sFlagTileBoundsInvalid = 0x01;
  static bool             sGlewInitialized;

  glm::dmat4 mWorldTransform;

  PlanetParameters mParams;
  LODVisitor       mLodVisitor;
  TileRenderer     mRenderer;

  TileSource*                mSrcDEM;
  TreeManager<RenderDataDEM> mTreeMgrDEM;

  TileSource*                mSrcIMG;
  TreeManager<RenderDataImg> mTreeMgrIMG;

  // global statistics
  double      mLastFrameClock;
  double      mSumFrameClock;
  std::size_t mSumDrawTiles;
  std::size_t mSumLoadTiles;

  std::size_t mMaxDrawTiles;
  std::size_t mMaxLoadTiles;

  glm::uint8 mFlags;
};
} // namespace csp::lodbodies
#endif // CSP_LOD_BODIES_VISTAPLANET_HPP
