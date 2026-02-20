////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "logger.hpp"
#include <iostream>


// ------------------------------------------------------------------------------------------------ //
// THE DOOR HANDLE (DLL EXPORTS)                                                                    //
// ------------------------------------------------------------------------------------------------ //

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::cesiumrenderer::Plugin;
}

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

// ------------------------------------------------------------------------------------------------ //
// THE IMPLEMENTATION                                                                               //
// ------------------------------------------------------------------------------------------------ //

namespace csp::cesiumrenderer {

void Plugin::init() {
  std::cout << ">>> CSP-CESIUM-RENDERER INIT CALLED <<<" << std::endl;
  logger().info("Loading plugin...");

  // Future code will go here

  logger().info("Loading done.");
}

void Plugin::deInit() {
  logger().info("Unloading plugin...");
    
  // Future cleanup code will go here

  logger().info("Unloading done.");
}

void Plugin::update() {
  // Runs 60 times a second. Do not put logger().info() here or it will freeze the console!
}

} // namespace csp::cesiumrenderer
