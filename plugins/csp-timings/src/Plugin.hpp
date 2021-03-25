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
/// timing statistics. This plugin can also be used to export recorded time series to CSV files.
class Plugin : public cs::core::PluginBase {
 public:
  void init() override;
  void deInit() override;
  void update() override;

 private:
  struct Range {
    Range(std::string name, uint32_t start, uint32_t end)
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
  std::vector<std::vector<std::vector<Range>>> mRecordedGPURanges;
  std::vector<std::vector<std::vector<Range>>> mRecordedCPURanges;

  int mFrameTimingConnection;
};

} // namespace csp::timings

#endif // CSP_TIMINGS_PLUGIN_HPP
