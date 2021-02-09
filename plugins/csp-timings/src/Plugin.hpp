////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_TIMINGS_PLUGIN_HPP
#define CSP_TIMINGS_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-gui/GuiItem.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"

#include <fstream>
#include <list>

namespace csp::timings {

/// A plugin which uses the built-in timer queries of CosmoScout VR to draw on-screen live frame
/// timing statistics. This plugin can also be used to export recorded time series to a CSV file.
class Plugin : public cs::core::PluginBase {
 public:
  void init() override;
  void deInit() override;
  void update() override;

 private:
  /// This store the statistics GUI element.
  std::unique_ptr<cs::gui::GuiItem> mGuiItem;

  bool mEnableRecording  = false;
  bool mEnableStatistics = false;
  std::list<std::unordered_map<std::string, cs::utils::FrameTimings::QueryResult>> mRecordedTimings;

  int mFrameTimingConnection;
};

} // namespace csp::timings

#endif // CSP_TIMINGS_PLUGIN_HPP
