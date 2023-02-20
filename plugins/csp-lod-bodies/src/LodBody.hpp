////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_LOD_PLANET_HPP
#define CSP_LOD_BODIES_LOD_PLANET_HPP

#include "../../../src/cs-graphics/Shadows.hpp"
#include "../../../src/cs-scene/CelestialSurface.hpp"
#include "../../../src/cs-scene/IntersectableObject.hpp"

#include "PlanetShader.hpp"
#include "TileSource.hpp"
#include "TileTextureArray.hpp"
#include "VistaPlanet.hpp"
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

#include <memory>

namespace cs::core {
class GraphicsEngine;
class GuiManager;
} // namespace cs::core

namespace csp::lodbodies {

/// An LodBody renders a planet from databases of hierarchical tiles. The tile data consists of
/// two components. Image data which determines the texture of the tiles and elevation data
/// (Digital Elevation Model or DEM) which determines the height map of each tile.
///
/// Each planet can make use of multiple data sources for image and elevation data. The user can
/// choose at runtime which data source should be used.
// DocTODO There probably are a thousand more things to explain.
class LodBody : public cs::scene::CelestialSurface,
                public cs::scene::IntersectableObject,
                public IVistaOpenGLDraw {
 public:
  LodBody(std::shared_ptr<cs::core::Settings>   settings,
      std::shared_ptr<cs::core::GraphicsEngine> graphicsEngine,
      std::shared_ptr<cs::core::SolarSystem>    solarSystem,
      std::shared_ptr<Plugin::Settings>         pluginSettings,
      std::shared_ptr<cs::core::GuiManager> pGuiManager, std::shared_ptr<GLResources> glResources);

  LodBody(LodBody const& other) = delete;
  LodBody(LodBody&& other)      = delete;

  LodBody& operator=(LodBody const& other) = delete;
  LodBody& operator=(LodBody&& other) = delete;

  ~LodBody() override;

  PlanetShader const& getShader() const;

  /// The planet is attached to this body.
  void               setObjectName(std::string objectName);
  std::string const& getObjectName() const;

  /// Sets the tile source for elevation data.
  void setDEMtileSource(std::shared_ptr<TileSource> source, uint32_t maxLevel);

  /// Gets the current tile source for elevation data.
  std::shared_ptr<TileSource> const& getDEMtileSource() const;

  /// Sets the tile source for image data.
  void setIMGtileSource(std::shared_ptr<TileSource> source, uint32_t maxLevel);

  /// Gets the current tile source for image data.
  std::shared_ptr<TileSource> const& getIMGtileSource() const;

  bool getIntersection(
      glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const override;
  double getHeight(glm::dvec2 lngLat) const override;

  void update();

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<cs::core::Settings>       mSettings;
  std::shared_ptr<cs::core::GraphicsEngine> mGraphicsEngine;
  std::shared_ptr<cs::core::SolarSystem>    mSolarSystem;
  std::shared_ptr<Plugin::Settings>         mPluginSettings;
  std::shared_ptr<cs::core::GuiManager>     mGuiManager;

  std::unique_ptr<VistaOpenGLNode>                 mGLNode;
  std::shared_ptr<TileSource>                      mDEMtileSource;
  std::shared_ptr<TileSource>                      mIMGtileSource;
  std::shared_ptr<cs::core::EclipseShadowReceiver> mEclipseShadowReceiver;

  std::string mObjectName;

  VistaPlanet  mPlanet;
  PlanetShader mShader;

  uint32_t mMaxLevelDEM = 0;
  uint32_t mMaxLevelIMG = 0;

  int mHeightScaleConnection = -1;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_LOD_PLANET_HPP
