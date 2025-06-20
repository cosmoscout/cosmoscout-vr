////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_SIMPLE_OBJECTS_SIMPLEOBJECT_HPP
#define CSP_SIMPLE_OBJECTS_SIMPLEOBJECT_HPP

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

namespace csp::simpleobjects {

/// A simple object on another celestial body.
class SimpleObject {
 public:

  /// name: Displayed name of the object
  /// config: Plugin::Settings::SimpleObject settings defined in .json
  SimpleObject(std::string const& name, Plugin::Settings::SimpleObject const& config,
      VistaSceneGraph* sceneGraph, std::shared_ptr<cs::core::Settings> settings,
      std::shared_ptr<cs::core::SolarSystem> solarSystem);
  
  SimpleObject(SimpleObject const& other) = delete;
  SimpleObject(SimpleObject&& other)      = default;

  SimpleObject& operator=(SimpleObject const& other) = delete;
  SimpleObject& operator=(SimpleObject&& other) = delete;

  ~SimpleObject();

  void update(/*double tTime, cs::scene::CelestialObserver const& oObs*/);

  void setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun);

  void updateConfig(std::string const& name, Plugin::Settings::SimpleObject const& config);

  std::string getName() const;

  // interface of scene::CelestialBody ---------------------------------------

  bool getIntersection(
      glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const;
  double getHeight(glm::dvec2 lngLat) const;
  
  void setEditEnabled(bool enabled);
  bool isEditedEnabled() const;

 private:
  std::shared_ptr<Plugin::Settings::SimpleObject>   mConfig;
  VistaSceneGraph*                                  mSceneGraph;
  std::shared_ptr<cs::core::Settings>               mSettings;
  std::shared_ptr<cs::core::SolarSystem>            mSolarSystem;
  std::shared_ptr<cs::scene::CelestialAnchorNode>   mAnchor;
  std::shared_ptr<cs::scene::CelestialBody>         mAnchorObject;
  std::shared_ptr<cs::graphics::GltfLoader>         mModel;
  std::shared_ptr<const cs::scene::CelestialObject> mSun;
  
  glm::dquat qRot;
  glm::dvec3 lastSurfaceNormal;
  
  bool editEnabled = false;
};


/*
/// A single satellite within the Solar System.
class Satellite : public cs::scene::CelestialBody {
 public:
  Satellite(Plugin::Settings::SimpleObject const& config, std::string const& anchorName,
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
}; */


} // namespace csp::simpleobjects

#endif // CSP_SIMPLE_OBJECTS_SIMPLEOBJECT_HPP
