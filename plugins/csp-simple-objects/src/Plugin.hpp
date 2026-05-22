////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_SIMPLE_OBJECTS_PLUGIN_HPP
#define CSP_SIMPLE_OBJECTS_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <regex>

namespace csp::simpleobjects {

class SimpleObject;

/// This plugin enables to place simpleobjects on celestial bodies in the Solar System.
/// The configuration of this plugin is done via the provided json config. See README.md for
/// details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {

    /// The settings for a simpleobject.
    struct SimpleObject {
      /// Path to the model. ".glb" and ".gltf" are allowed formats.
      std::string mModelFile;
      /// Path to the environment map. ".dds", ".ktx" and ".kmg" are allowed formats.
      std::string mEnvironmentMap;
      std::string mAnchorName;
      glm::dvec2 mLngLat;
      cs::utils::DefaultProperty<glm::dquat> mRotation{glm::dquat(1.0,0.0,0.0,0.0)};
      cs::utils::DefaultProperty<bool> mAlignToSurface{false};
      cs::utils::DefaultProperty<double> mElevation{0.0};
      cs::utils::DefaultProperty<double> mScale{1.0};
      cs::utils::DefaultProperty<double> mDiagonalLength{5.0};
    };

    std::map<std::string, SimpleObject> mSimpleObjects;
  };

  void init() override;
  void deInit() override;
  void update() override;


 private:
  Settings                                   mPluginSettings;
  std::vector<std::shared_ptr<SimpleObject>> mSimpleObjects;
  std::shared_ptr<SimpleObject> tmpSimpleObject;
  std::shared_ptr<double> minVisibilityAngle;

  bool pickLocationToolEnabled = false;
  int mOnClickConnection       = -1;
  int mOnObjectAddedConnection = -1;
  int mOnObjectRemovedConnection = -1;

  std::string modelFile = "";
  std::string environmentMap = "";
  bool alignToSurfaceEnabled   = false;

  void initDropdown(const std::string callbackName, const std::string folder, std::regex pattern);
  void addObject(std::string name, Settings::SimpleObject settings);
};

} // namespace csp::simpleobjects

#endif // CSP_SIMPLE_OBJECTS_PLUGIN_HPP
