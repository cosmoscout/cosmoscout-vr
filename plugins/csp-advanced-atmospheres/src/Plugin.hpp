////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ADVANCED_ATMOSPHERE_PLUGIN_HPP
#define CSP_ADVANCED_ATMOSPHERE_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"

class VistaOpenGLNode;

namespace csp::advanced_atmospheres {

class Atmosphere;

/// This plugin adds atmospheres to planets and moons. It uses mie and rayleigh scattering for
/// rendering atmospheric effects. It is configurable via the application config file. See README.md
/// for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    struct Atmosphere {};

    std::unordered_map<std::string, Atmosphere> mAtmospheres;

    cs::utils::DefaultProperty<bool> mEnabled{true};
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

} // namespace csp::advanced_atmospheres

#endif // CSP_ADVANCED_ATMOSPHERE_PLUGIN_HPP
