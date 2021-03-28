////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_GUIDE_PLUGIN_HPP
#define CSP_GUIDE_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"

#include <list>
#include <string>
#include <vector>

namespace csp::guide {

/// This plugin adds a guide which helps the user explore the application.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    
  };

  void init() override;
  void update() override;
  void deInit() override;

 private:
  void onLoad();
  void unload(Settings const& pluginSettings);

  Settings mPluginSettings;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::guide

#endif // CSP_GUIDE_PLUGIN_HPP
