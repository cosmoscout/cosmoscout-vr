////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_COORDINATE_ARROWS_PLUGIN_HPP
#define CSP_COORDINATE_ARROWS_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <optional>
#include <memory>
#include <map>

namespace csp::coordinatearrows {

class Arrow;

/// Your plugin description here!
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    // Settings for a group of arrows.
    struct Arrows {};

    // All groups of arrows with their name as key.
    std::map<std::string, Arrows> mArrows;

    // Enables/Disables the all arrows.
    cs::utils::DefaultProperty<bool> mEnableArrows{true};
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();
  std::map<std::string, std::vector<std::shared_ptr<Arrow>>> mArrows;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::coordinatearrows

#endif // CSP_COORDINATE_ARROWS_PLUGIN_HPP
