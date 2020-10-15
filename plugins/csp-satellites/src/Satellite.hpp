////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_SATELLITE_HPP
#define CS_CORE_SATELLITE_HPP

#include "Plugin.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-scene/CelestialBody.hpp"

#include "../../../src/cs-scene/CelestialAnchorNode.hpp"
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
class Satellite : public cs::scene::CelestialBody {
 public:
  Satellite(Plugin::Settings::Satellite const& config, std::string const& anchorName,
      VistaSceneGraph* sceneGraph, std::shared_ptr<cs::core::Settings> settings,
      std::shared_ptr<cs::core::SolarSystem> solarSystem);

  Satellite(Satellite const& other) = delete;
  Satellite(Satellite&& other)      = default;

  Satellite& operator=(Satellite const& other) = delete;
  Satellite& operator=(Satellite&& other) = delete;

  ~Satellite() override;

  void update(double tTime, cs::scene::CelestialObserver const& oObs) override;

  void setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun);

  // interface of scene::CelestialBody ---------------------------------------

  bool getIntersection(
      glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const override;
  double getHeight(glm::dvec2 lngLat) const override;

 private:
  VistaSceneGraph*                                  mSceneGraph;
  std::shared_ptr<cs::core::Settings>               mSettings;
  std::shared_ptr<cs::core::SolarSystem>            mSolarSystem;
  std::unique_ptr<VistaTransformNode>               mAnchor;
  std::unique_ptr<cs::graphics::GltfLoader>         mModel;
  std::shared_ptr<const cs::scene::CelestialObject> mSun;
};
} // namespace csp::satellites

#endif // CS_CORE_SATELLITE_HPP
