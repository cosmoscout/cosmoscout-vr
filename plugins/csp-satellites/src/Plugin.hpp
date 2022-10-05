////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_SATELLITES_PLUGIN_HPP
#define CSP_SATELLITES_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

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

    /// The settings for a satellite.
    struct Satellite {
      /// Path to the model. ".glb" and ".gltf" are allowed formats.
      std::string mModelFile;

      /// Path to the environment map. ".dds", ".ktx" and ".kmg" are allowed formats.
      std::string mEnvironmentMap;
    };

    std::map<std::string, Satellite> mSatellites;
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();
  void onSave();

  Settings                                mPluginSettings;
  std::vector<std::shared_ptr<Satellite>> mSatellites;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::satellites

#endif // CSP_SATELLITES_PLUGIN_HPP
