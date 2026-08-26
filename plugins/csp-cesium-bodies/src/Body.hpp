////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_BODIES_BODY_HPP
#define CSP_CESIUM_BODIES_BODY_HPP

#include <cstdint>
#include <memory>
#include <string>

namespace cs {
namespace core {
class SolarSystem;
class Settings;
}

namespace scene {
class CelestialSurface;
class CelestialObject;
class CelestialObserver;
} // namespace scene
} // namespace cs

namespace Cesium3DTilesSelection {
class Tileset;
struct TilesetOptions;
class TilesetExternals;
} // namespace Cesium3DTilesSelection

namespace csp::cesiumbodies {

class TilesetRenderer;

class Body {
 public:
  Body(std::string const& name, Cesium3DTilesSelection::TilesetExternals const& tilesetExternals,
      int64_t assetId, std::string const& ionToken,
      Cesium3DTilesSelection::TilesetOptions const& options,
      std::shared_ptr<cs::core::SolarSystem>        solarSystem,
      std::shared_ptr<cs::core::Settings>     settings);

  ~Body();

  void update(cs::scene::CelestialObserver& observer);

  std::shared_ptr<cs::scene::CelestialSurface> getSurface() const;

 private:
  std::string mName;

  std::shared_ptr<const cs::scene::CelestialObject> mCelestialObject;

  std::unique_ptr<Cesium3DTilesSelection::Tileset> mTileset;
  std::shared_ptr<TilesetRenderer>                 mTilesetRenderer;
};
} // namespace csp::cesiumbodies

#endif // CSP_CESIUM_BODIES_BODY_HPP
