////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_LOD_PLANET_HPP
#define CSP_LOD_BODIES_LOD_PLANET_HPP

#include "../../../src/cs-graphics/Shadows.hpp"
#include "../../../src/cs-scene/CelestialBody.hpp"

#include "PlanetShader.hpp"
#include "TileSource.hpp"
#include "TileTextureArray.hpp"
#include "VistaPlanet.hpp"
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

#include <memory>

namespace cs::scene {
class CelestialAnchorNode;
}

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
class LodBody : public cs::scene::CelestialBody, public IVistaOpenGLDraw {
 public:
  LodBody(std::shared_ptr<cs::core::Settings> const& settings,
      std::shared_ptr<cs::core::GraphicsEngine>      graphicsEngine,
      std::shared_ptr<cs::core::SolarSystem>         solarSystem,
      std::shared_ptr<Plugin::Settings> const&       pluginSettings,
      std::shared_ptr<cs::core::GuiManager> const& pGuiManager, std::string const& sCenterName,
      std::string const& sFrameName, std::shared_ptr<GLResources> const& glResources,
      double tStartExistence, double tEndExistence);

  LodBody(LodBody const& other) = delete;
  LodBody(LodBody&& other)      = delete;

  LodBody& operator=(LodBody const& other) = delete;
  LodBody& operator=(LodBody&& other) = delete;

  ~LodBody() override;

  PlanetShader const& getShader() const;

  void setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun);

  /// Sets the tile source for elevation data.
  void setDEMtileSource(std::shared_ptr<TileSource> source);

  /// Gets the current tile source for elevation data.
  std::shared_ptr<TileSource> const& getDEMtileSource() const;

  /// Sets the tile source for image data.
  void setIMGtileSource(std::shared_ptr<TileSource> source);

  /// Gets the current tile source for image data.
  std::shared_ptr<TileSource> const& getIMGtileSource() const;

  bool getIntersection(
      glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const override;
  double     getHeight(glm::dvec2 lngLat) const override;
  glm::dvec3 getRadii() const override;

  void update(double tTime, cs::scene::CelestialObserver const& oObs) override;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<cs::core::Settings>               mSettings;
  std::shared_ptr<cs::core::GraphicsEngine>         mGraphicsEngine;
  std::shared_ptr<cs::core::SolarSystem>            mSolarSystem;
  std::shared_ptr<Plugin::Settings>                 mPluginSettings;
  std::shared_ptr<const cs::scene::CelestialObject> mSun;
  std::shared_ptr<cs::core::GuiManager>             mGuiManager;

  std::unique_ptr<VistaOpenGLNode> mGLNode;
  std::shared_ptr<TileSource>      mDEMtileSource;
  std::shared_ptr<TileSource>      mIMGtileSource;

  VistaPlanet  mPlanet;
  PlanetShader mShader;
  glm::dvec3   mRadii;
  int          mHeightScaleConnection = -1;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_LOD_PLANET_HPP
