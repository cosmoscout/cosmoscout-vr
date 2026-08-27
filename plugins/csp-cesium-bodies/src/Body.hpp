////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_BODIES_BODY_HPP
#define CSP_CESIUM_BODIES_BODY_HPP

#include "../../../src/cs-scene/CelestialSurface.hpp"
#include "../../../src/cs-scene/IntersectableObject.hpp"

#include <Cesium3DTilesSelection/Tile.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <memory>
#include <string>

namespace cs {
namespace core {
class SolarSystem;
class Settings;
} // namespace core

namespace scene {
class CelestialObject;
class CelestialObserver;
} // namespace scene
} // namespace cs

namespace Cesium3DTilesSelection {
class Tile;
class Tileset;
struct TilesetOptions;
class TilesetExternals;
} // namespace Cesium3DTilesSelection

namespace csp::cesiumbodies {

class TilesetRenderer;

class Body : public cs::scene::CelestialSurface, public cs::scene::IntersectableObject {
 public:
  Body(std::string const& name, Cesium3DTilesSelection::TilesetExternals const& tilesetExternals,
      int64_t assetId, std::string const& ionToken,
      Cesium3DTilesSelection::TilesetOptions const& options,
      std::shared_ptr<cs::core::SolarSystem>        solarSystem,
      std::shared_ptr<cs::core::Settings>           settings);

  ~Body() override;

  void update(cs::scene::CelestialObserver& observer);

  double getHeight(glm::dvec2 lngLat) const override;

  bool getIntersection(
      glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const override;

 private:
  std::string mName;

  std::shared_ptr<cs::core::SolarSystem>            mSolarSystem;
  std::shared_ptr<const cs::scene::CelestialObject> mCelestialObject;

  std::unique_ptr<Cesium3DTilesSelection::Tileset>   mTileset;
  std::shared_ptr<TilesetRenderer>                   mTilesetRenderer;

  /// We cache the last height tile to avoid unnecessary tile traversal.
  mutable Cesium3DTilesSelection::Tile::ConstPointer mLastHeightTile;
};

} // namespace csp::cesiumbodies

#endif // CSP_CESIUM_BODIES_BODY_HPP
