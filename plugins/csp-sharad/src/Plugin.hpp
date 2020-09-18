////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_SHARAD_PLUGIN_HPP
#define CSP_SHARAD_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "Sharad.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>

namespace csp::sharad {

/// This plugin allows the display of mars subsurface layers captured by the Mars Reconnaissance
/// Orbiter. The configuration is done via the applications config file. See README.md for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    std::string                      mAnchor;
    cs::utils::Property<std::string> mFilePath;
    cs::utils::DefaultProperty<bool> mEnabled{false};
  };

  void init() override;
  void deInit() override;

 private:
  void onLoad();

  Settings                                      mPluginSettings;
  std::vector<std::shared_ptr<Sharad>>          mSharads;
  std::vector<std::unique_ptr<VistaOpenGLNode>> mSharadNodes;

  int mActiveBodyConnection = -1;
  int mOnLoadConnection     = -1;
  int mOnSaveConnection     = -1;
};

} // namespace csp::sharad

#endif // CSP_SHARAD_PLUGIN_HPP
