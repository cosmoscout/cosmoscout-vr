////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_PLUGIN_HPP
#define CSP_ATMOSPHERES_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"

class VistaOpenGLNode;

namespace csp::atmospheres {

class Atmosphere;

/// This plugin adds atmospheres to planets and moons. It uses mie and rayleigh scattering for
/// rendering atmospheric effects. It is configurable via the application config file. See README.md
/// for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    struct Atmosphere {
      enum class Model { eCosmoScoutVR, eBruneton };

      double                            mHeight; ///< In meters.
      cs::utils::DefaultProperty<Model> mModel{Model::eCosmoScoutVR};
      nlohmann::json                    mModelSettings;

      cs::utils::DefaultProperty<bool>  mEnableWater{false};
      cs::utils::DefaultProperty<float> mWaterLevel{0.F}; ///< In meters.
      cs::utils::DefaultProperty<bool>  mEnableClouds{true};
      std::optional<std::string>        mCloudTexture;          ///< Path to the cloud texture.
      cs::utils::DefaultProperty<float> mCloudAltitude{3000.F}; ///< In meters.
      cs::utils::DefaultProperty<bool>  mEnableLightShafts{false};
    };

    cs::utils::DefaultProperty<bool>            mEnabled{true};
    std::unordered_map<std::string, Atmosphere> mAtmospheres;
  };

  void init() override;
  void deInit() override;

  void update() override;

 private:
  void onLoad();
  void onSave();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();
  std::unordered_map<std::string, std::shared_ptr<Atmosphere>> mAtmospheres;

  int mEnableHDRConnection = -1;
  int mOnLoadConnection    = -1;
  int mOnSaveConnection    = -1;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERES_PLUGIN_HPP
