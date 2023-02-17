////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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
class VistaPlanet : public cs::graphics::ShadowCaster {
 public:
  explicit VistaPlanet(std::shared_ptr<GLResources> glResources, uint32_t tileResolution);

  VistaPlanet(VistaPlanet const& other) = delete;
  VistaPlanet(VistaPlanet&& other)      = delete;

  VistaPlanet& operator=(VistaPlanet const& other) = delete;
  VistaPlanet& operator=(VistaPlanet&& other) = delete;

  ~VistaPlanet();

  void draw();
  void drawForShadowMap() override;

  void       setWorldTransform(glm::dmat4 const& mat);
  glm::dmat4 getWorldTransform() const;

  void setEnabled(bool enabled);
  bool getEnabled() const;

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

  /// Set planet radii. This is a potentially expensive operation since it invalidates
  /// the cached bounding volume for all tiles and requires recalculating them.
  void              setRadii(glm::dvec3 const& radii);
  glm::dvec3 const& getRadii() const;

  /// Set factor by which to scale height data. This is a potentially expensive operation since it
  /// invalidates the cached bounding volume for all tiles and requires recalculating them.
  void   setHeightScale(float scale);
  double getHeightScale() const;

  /// This will affect the average size of the rendered tiles when projected onto the screen. A
  /// higher lodFactor will reduce in smaller tiles and hence in a higher data density.
  void   setLODFactor(float lodFactor);
  double getLODFactor() const;

  /// The tile quadtrees will always be refined at least up to this level.
  void setMinLevel(int minLevel);
  int  getMinLevel() const;

  /// The tile quadtrees will be refined at most up to this level.
  void setMaxLevel(int maxLevel);
  int  getMaxLevel() const;

  /// Returns the TileRenderer instance used to render this VistaPlanet.
  TileRenderer&       getTileRenderer();
  TileRenderer const& getTileRenderer() const;

  /// Returns the LODVisitor instance used to determine which tiles to render and load.
  LODVisitor&       getLODVisitor();
  LODVisitor const& getLODVisitor() const;

 private:
  void updateStatistics(int frameCount);
  void updateTileBounds();
  void updateTileTrees(int frameCount);
  void traverseTileTrees(int frameCount, glm::dmat4 const& matM, glm::mat4 const& matV,
      glm::mat4 const& matP, glm::ivec4 const& viewport);
  void processLoadRequests();
  void renderTiles(int frameCount, glm::dmat4 const& matM, glm::mat4 const& matV,
      glm::mat4 const& matP, cs::graphics::ShadowMap* shadowMap);

  glm::mat4         getViewMatrix() const;
  static glm::mat4  getProjectionMatrix();
  static glm::ivec4 getViewport();

  static glm::uint8 const sFlagTileBoundsInvalid = 0x01;
  static bool             sGlewInitialized;

  glm::dmat4 mWorldTransform;
  bool       mEnabled = false;

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
