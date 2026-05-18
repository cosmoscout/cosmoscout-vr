////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_EFFECTS_PLUGIN_HPP
#define CSP_VISUAL_EFFECTS_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace csp::visualeffects {

class SolarFlares;

/// Your plugin description here!
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    struct SolarFlares {};
    std::unordered_map<std::string, SolarFlares> mSolarFlares;
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();
  std::unordered_map<std::string, std::shared_ptr<SolarFlares>> mSolarFlares;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::visualeffects

#endif // CSP_VISUAL_EFFECTS_PLUGIN_HPP
