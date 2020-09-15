////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_SATELLITES_PLUGIN_HPP
#define CSP_SATELLITES_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace csp::satellites {

class Satellite;

/// This plugin enables to place satellites into the Solar System.
/// The configuration of this plugin is done via the provided json config. See README.md for
/// details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    /// The initial transformation of the satellite.
    struct Transformation {
      glm::dvec3 mTranslation;
      glm::dquat mRotation;
      double     mScale;
    };

    /// The settings for a satellite.
    struct Satellite {
      std::string mModelFile;      ///< Path to the model. ".glb" and ".gltf" are
                                   ///< allowed formats.
      std::string mEnvironmentMap; ///< Path to the environment map. ".dds",
                                   ///< ".ktx" and ".kmg" are allowed formats.
      std::optional<Transformation> mTransformation;
    };

    std::map<std::string, Satellite> mSatellites;
  };

  void init() override;
  void deInit() override;

 private:
  Settings                                mPluginSettings;
  std::vector<std::shared_ptr<Satellite>> mSatellites;
};

} // namespace csp::satellites

#endif // CSP_SATELLITES_PLUGIN_HPP
