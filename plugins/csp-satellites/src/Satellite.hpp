////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_SATELLITE_HPP
#define CS_CORE_SATELLITE_HPP

#include "Plugin.hpp"

#include "../../../src/cs-core/Settings.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>

namespace cs::graphics {
class GltfLoader;
} // namespace cs::graphics

namespace cs::core {
class Settings;
class SolarSystem;
} // namespace cs::core

class VistaTransformNode;

namespace csp::satellites {

/// A single satellite within the Solar System.
class Satellite {
 public:
  Satellite(Plugin::Settings::Satellite const& config, std::string objectName,
      VistaSceneGraph* sceneGraph, std::shared_ptr<cs::core::Settings> settings,
      std::shared_ptr<cs::core::SolarSystem> solarSystem);

  Satellite(Satellite const& other) = delete;
  Satellite(Satellite&& other)      = default;

  Satellite& operator=(Satellite const& other) = delete;
  Satellite& operator=(Satellite&& other)      = delete;

  ~Satellite();

  void update();

 private:
  VistaSceneGraph*                          mSceneGraph;
  std::shared_ptr<cs::core::Settings>       mSettings;
  std::shared_ptr<cs::core::SolarSystem>    mSolarSystem;
  std::unique_ptr<VistaTransformNode>       mAnchor;
  std::unique_ptr<cs::graphics::GltfLoader> mModel;

  std::string mObjectName;
};
} // namespace csp::satellites

#endif // CS_CORE_SATELLITE_HPP
