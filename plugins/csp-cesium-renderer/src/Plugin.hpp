////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_RENDERER_PLUGIN_HPP
#define CSP_CESIUM_RENDERER_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "CesiumTilesetRenderer.hpp"
#include <Cesium3DTilesSelection/Tileset.h>
#include <memory>
#include <optional>
#include <string>

namespace CesiumAsync {
class AsyncSystem;
}
namespace CesiumUtility {
class CreditSystem;
}

namespace csp::cesiumrenderer {

class Plugin : public cs::core::PluginBase {
 public:
  /// All fields are std::optional — if omitted from JSON, the hardcoded
  /// defaults are used. This means "csp-cesium-renderer": {} keeps working.
  struct Settings {
    std::optional<int64_t>     mIonAssetId;             ///< Cesium Ion asset ID (default: 2275207)
    std::optional<std::string> mIonToken;               ///< Cesium Ion access token
    std::optional<int64_t>     mCacheSizeMB;            ///< Tile cache limit in MB (default: 256)
    std::optional<int32_t>     mMaxConcurrentDownloads; ///< Concurrent tile downloads (default: 20)
    std::optional<double>      mMaxScreenSpaceError;    ///< LOD SSE threshold px (default: 16.0)
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  Settings                                         mPluginSettings;
  std::shared_ptr<CesiumAsync::AsyncSystem>        mAsyncSystem;
  std::shared_ptr<CesiumUtility::CreditSystem>     mCreditSystem;
  std::unique_ptr<Cesium3DTilesSelection::Tileset> mTileset;
  std::shared_ptr<CesiumTilesetRenderer>           mRenderer;
};

} // namespace csp::cesiumrenderer

#endif // CSP_CESIUM_RENDERER_PLUGIN_HPP