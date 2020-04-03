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
#include <chrono>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace cs::utils {

class TimerQueryPool;

/// Responsible for measuring time. It is possible to measure either or both CPU and GPU time. To
/// create a time measuring object use ScopedTimer. This class is not thread-safe, so you should
/// only use it to measure time on the main thread. You should not need to instantiate this class.
/// One instance will be created by the application class and is passed to each and every plugin.
class CS_UTILS_EXPORT FrameTimings {
 public:
  /// Defines which timings should be measured.
  enum class QueryMode {
    eCPU, ///< Only the CPU time will be measured.
    eGPU, ///< Only the GPU time will be measured.
    eBoth ///< CPU and GPU time will be measured.
  };

  /// To enable or disable time measuring globally.
  Property<bool> pEnableMeasurements = false;

  /// Get the last frame time in milliseconds. This is the maximum of CPU and GPU time, excluding
  /// any waiting for vertical synchronization.
  Property<double> pFrameTime = 0.0;

  /// A ScopedTimer is responsible for measuring time for its entire existence. The timer will
  /// start measuring upon creation and stop measuring on deletion. If multiple timers with the same
  /// name are created during one frame, their timings will be accumulated.
  class CS_UTILS_EXPORT ScopedTimer {
   public:
    /// @param name The name of the measured time.
    /// @param mode The mode of querying. See QueryMode for more info.
    explicit ScopedTimer(std::string name, QueryMode mode = QueryMode::eBoth);

    ScopedTimer(ScopedTimer const& other) = delete;
    ScopedTimer(ScopedTimer&& other)      = delete;

    ScopedTimer& operator=(ScopedTimer const& other) = delete;
    ScopedTimer& operator=(ScopedTimer&& other) = delete;

    ~ScopedTimer();

   private:
    std::string mName;
  };

  struct CS_UTILS_EXPORT QueryResult {
    uint64_t mGPUTime = 0; ///< in ns (==10^-9 s)
    uint64_t mCPUTime = 0; ///< in ns (==10^-9 s)
  };

  /// You should not need to instantiate this class. One instance will be created by the application
  /// class and is passed to each and every plugin.
  explicit FrameTimings();

  FrameTimings(FrameTimings const& other) = delete;
  FrameTimings(FrameTimings&& other)      = delete;

  FrameTimings& operator=(FrameTimings const& other) = delete;
  FrameTimings& operator=(FrameTimings&& other) = delete;

  ~FrameTimings() = default;

  /// Starts a timer with the given name and mode. You can use this interface, however the
  /// ScopedTimer is often more easy to use.
  static void start(std::string const& name, QueryMode mode = QueryMode::eBoth);

  /// Stops the timer with the given name and saves the results. You can use this interface, however
  /// the ScopedTimer is often more easy to use.
  static void end(std::string const& name);

  /// Ends the timer that was started last. You can use this interface, however the ScopedTimer is
  /// often more easy to use.
  static void end();

  /// Starts the time measurement for the current frame. No need to call this manually. The
  /// application is responsible for this.
  void startFullFrameTiming();

  /// Ends the time measurement for the current frame. No need to call this manually. The
  /// application is responsible for this.
  void endFullFrameTiming();

  /// Updates the frame timings. The application takes care of calling this on the primary instance.
  void update() const;

  /// Once update() has been called, QueryResults most likely are available. However, we will only
  /// get results from the last frame in order to prevent any blocking. Sometimes it might also take
  /// a few frames longer to get any results from the GPU.
  std::unordered_map<std::string, QueryResult> getCalculatedQueryResults() const;

 private:
  int                                            mCurrentIndex = 0;
  std::array<std::shared_ptr<TimerQueryPool>, 2> mFullFrameTimerPools;
};

/// The TimerQueryPool is used in a double-buffer fashion internally by the FrameTimings class. You
/// will not need to use this class directly.
class CS_UTILS_EXPORT TimerQueryPool {
 public:
  explicit TimerQueryPool(std::size_t max_size);

  /// Do not try to copy this class!
  TimerQueryPool(TimerQueryPool const&) = delete;
  void operator=(TimerQueryPool const&) = delete;

  TimerQueryPool(TimerQueryPool&&) = delete;
  void operator=(TimerQueryPool&&) = delete;

  ~TimerQueryPool() = default;

  void start(std::string const& name, FrameTimings::QueryMode mode);
  void end(std::string const& name);

  std::optional<std::size_t> timestamp();

  /// Fetch timestamps from GPU and calculate time diffs.
  void calculateQueryResults();

  [[nodiscard]] std::unordered_map<std::string, std::vector<FrameTimings::QueryResult>> const&
  getQueryResults() const;

 private:
  struct QueryRange {
    std::size_t                                    mGPUStart = 0;
    std::size_t                                    mGPUEnd   = 0;
    std::chrono::high_resolution_clock::time_point mCPUStart;
    std::chrono::high_resolution_clock::time_point mCPUEnd;
    FrameTimings::QueryMode                        mMode{};
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
