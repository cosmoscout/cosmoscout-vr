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

/// This plugin allows basic capturing of high-quality videos using csp-web-api. 'Basic' means that
/// (for now) only the observer transformation, the simulation time and the exposure of the HDR mode
/// is captured. This however, can be changed in the future.
/// Capturing works in two phases: First, the user navigates through space while 'recording'. This
/// produces a python script in the bin/ directory which can be executed to capture the individual
/// frame using csp-web-api. This two-step approach has the advantage that recording can be done at
/// high frame rates (with all settings reduced to the bare minimum) while capturing can be done at
/// high resolution and high quality.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    /// This should be set to the port of csp-web-api.
    cs::utils::Property<uint16_t> mWebAPIPort;

    /// These can be toggled via the user interface.
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
