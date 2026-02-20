////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_RENDERER_PLUGIN_HPP
#define CSP_CESIUM_RENDERER_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"

namespace csp::cesiumrenderer {

class Plugin : public cs::core::PluginBase {
public:
  void init() override;
void deInit() override;
void update() override;
  
};

} // namespace csp::cesiumrenderer

#endif // CSP_CESIUM_RENDERER_PLUGIN_HPP