////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_RENDERER_PLUGIN_HPP
#define CSP_CESIUM_RENDERER_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include <memory>  
#include <Cesium3DTilesSelection/Tileset.h>
#include "CesiumTilesetRenderer.hpp"


namespace CesiumAsync {
class AsyncSystem;
}
namespace CesiumUtility {
class CreditSystem;
}

namespace csp::cesiumrenderer {

class Plugin : public cs::core::PluginBase {
public:
  void init() override;
  void deInit() override;
  void update() override;

private:
  std::shared_ptr<CesiumAsync::AsyncSystem>    mAsyncSystem;
  std::shared_ptr<CesiumUtility::CreditSystem> mCreditSystem;
  std::unique_ptr<Cesium3DTilesSelection::Tileset> mTileset;
  std::unique_ptr<CesiumTilesetRenderer> mRenderer;
  
};

} // namespace csp::cesiumrenderer

#endif // CSP_CESIUM_RENDERER_PLUGIN_HPP