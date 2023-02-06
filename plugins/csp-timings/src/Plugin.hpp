////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_TIMINGS_PLUGIN_HPP
#define CSP_TIMINGS_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-gui/GuiItem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"

#include <fstream>
#include <list>

namespace csp::timings {

/// A plugin which uses the built-in timer queries of CosmoScout VR to draw on-screen live frame
/// timing statistics. This plugin can also be used to export recorded time series to CSV files.
class Plugin : public cs::core::PluginBase {
 public:
  void init() override;
  void deInit() override;
  void update() override;

 private:
  struct TimerRange {
    TimerRange(std::string name, uint32_t start, uint32_t end)
        : mName(std::move(name))
        , mStart(start)
        , mEnd(end) {
    }

    std::string mName;

    // Frame-relative timestamps in microseconds.
    uint32_t mStart;
    uint32_t mEnd;
  };

  /// This store the statistics GUI element.
  std::unique_ptr<cs::gui::GuiItem> mGuiItem;

  bool mEnableRecording  = false;
  bool mEnableStatistics = false;

  /// Stores recorded ranges for the GPU and CPU for each frame for each nesting level. So the
  /// outer-most vector is per recorded frame, the middle per range nesting level and the inner-most
  /// contains all ranges for the specific level.
  /// Frame Indices | Nesting Levels | Ranges
  std::vector<std::vector<std::vector<TimerRange>>> mRecordedGPURanges;
  std::vector<std::vector<std::vector<TimerRange>>> mRecordedCPURanges;

  /// Sample queries do not support nesting.
  std::vector<int64_t> mTimestamps;

  int mFrameTimingConnection;
};

} // namespace csp::timings

#endif // CSP_TIMINGS_PLUGIN_HPP
