////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_RECORDER_PLUGIN_HPP
#define CSP_RECORDER_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"

#include <fstream>

namespace csp::recorder {

class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    cs::utils::DefaultProperty<bool> mRecordObserver{true};
    cs::utils::DefaultProperty<bool> mRecordTime{true};
    cs::utils::DefaultProperty<bool> mRecordExposure{false};
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();

  Settings mPluginSettings;

  bool          mRecording = false;
  std::ofstream mOutFile;
  uint32_t      mFrameCounter = 0;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::recorder

#endif // CSP_RECORDER_PLUGIN_HPP
