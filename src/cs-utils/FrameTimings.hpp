////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_FRAME_TIMINGS_HPP
#define CS_UTILS_FRAME_TIMINGS_HPP

#include "Property.hpp"
#include "cs_utils_export.hpp"

#include <array>
#include <boost/optional.hpp>
#include <chrono>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cs::utils {

class TimerQueryPool;

/// Responsible for measuring time. It is possible to measure either or both CPU and GPU time. To
/// create a time measuring object use ScopedTimer.
class CS_UTILS_EXPORT FrameTimings {
 public:
  /// Defines which timings should be measured.
  enum class QueryMode {
    eCPU, ///< Only the CPU time will be measured.
    eGPU, ///< Only the GPU time will be measured.
    eBoth ///< CPU and GPU time will be measured.
  };

  /// To enable or disable time measuring.
  Property<bool> pEnableMeasurements = false;

  /// Get the last frame time. DocTODO for the whole frame?
  Property<double> pFrameTime = 0.0;

  /// A ScopedTimer is responsible for measuring time for its entire existence. The timer will
  /// start measuring upon creation and stop measuring on deletion.
  class CS_UTILS_EXPORT ScopedTimer {
   public:
    /// @param name The name of the measured time.
    /// @param mode The mode of querying. See QueryMode for more info.
    explicit ScopedTimer(std::string const& name, QueryMode mode = QueryMode::eBoth);
    ~ScopedTimer();

   private:
    std::string mName;
  };

  struct CS_UTILS_EXPORT QueryResult {
    uint64_t mGPUTime = 0; ///< in ns (==10^-9 s)
    uint64_t mCPUTime = 0; ///< in ns (==10^-9 s)
  };

  /// Starts a timer with the given name and mode.
  static void start(std::string const& name, QueryMode mode = QueryMode::eBoth);

  /// Stops the timer with the given name and saves the results.
  static void end(std::string const& name);

  /// Ends the timer that was started last.
  static void end();

  /// Returns the results of all measurements. (DocTODO in the last frame?)
  std::unordered_map<std::string, QueryResult> getCalculatedQueryResults();

  explicit FrameTimings();
  ~FrameTimings() = default;

  /// Starts the time measurement for the current frame. No need to call this manually. The
  /// application is responsible for this.
  void startFullFrameTiming();

  /// Ends the time measurement for the current frame. No need to call this manually. The
  /// application is responsible for this.
  void endFullFrameTiming();

  /// Updates the frame timings. The application takes care of calling this.
  void update();

 private:
  int                                            mCurrentIndex = 0;
  std::array<std::shared_ptr<TimerQueryPool>, 2> mFullFrameTimerPools;
};

/// DocTODO
class CS_UTILS_EXPORT TimerQueryPool {
 public:
  explicit TimerQueryPool(std::size_t max_size);

  TimerQueryPool(TimerQueryPool const&) = delete;
  void operator=(TimerQueryPool const&) = delete;

  void start(std::string const& name, FrameTimings::QueryMode mode);
  void end(std::string const& name);

  boost::optional<std::size_t> timestamp();

  /// Fetch timestamps from gpu and calculate time diffs.
  void calculateQueryResults();

  std::unordered_map<std::string, std::vector<FrameTimings::QueryResult>> const&
  getQueryResults() const;

 private:
  struct QueryRange {
    std::size_t                                    mGPUStart = 0;
    std::size_t                                    mGPUEnd   = 0;
    std::chrono::high_resolution_clock::time_point mCPUStart;
    std::chrono::high_resolution_clock::time_point mCPUEnd;
    FrameTimings::QueryMode                        mMode;
  };

  std::size_t           mMaxSize;
  std::vector<uint32_t> mQueries;
  std::vector<uint64_t> mTimestamps;
  int32_t               mQueryDone;
  std::size_t           mIndex;

  std::unordered_map<std::string, std::vector<QueryRange>>                mQueryRanges;
  std::unordered_map<std::string, std::vector<FrameTimings::QueryResult>> mQueryResults;
};

} // namespace cs::utils

#endif // CS_UTILS_FRAME_TIMINGS_HPP
