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

class TimerPool;

/// This is a singleton class which can be used in the main thread to record CPU und GPU timings.
/// This class actively supports measuring nested ranges so it's perfectively fine to start ranges
/// for large parts of the code and multiple smaller subranges for smaller parts. To measure the
/// time spent on the CPU or the GPU by a specific block of code, simply create a
/// FrameTimings::ScopedTimer. This will start a measuring range in its constructor and and end the
/// range in its destructor.
class CS_UTILS_EXPORT FrameTimings {
 public:
  /// Defines which timings should be measured.
  enum class QueryMode {
    eCPU, ///< Only the CPU time will be measured.
    eGPU, ///< Only the GPU time will be measured.
    eBoth ///< CPU and GPU time will be measured.
  };

  /// This struct contains information on one specific timing range. It is used internally by the
  /// FrameTimings singleton and can be accessed via its getRanges() method.
  struct Range {

    /// The name of the range as it was passed to the constructor of the ScopedTimer or the
    /// FrameTimings::startRange() method.
    std::string mName;

    /// This contains the number of timing ranges which were active when this range was started.
    uint32_t mNestingLevel{};

    /// Timestamps in nanoseconds when the range started / ended on the CPU / GPU. They will be
    /// filled with data only if the QueryMode was set accordingly.
    int64_t mCPUStart{};
    int64_t mCPUEnd{};
    int64_t mGPUStart{};
    int64_t mGPUEnd{};

    /// The mode this range was started with.
    QueryMode mMode;

    /// These are used internally by the FrameTimings class to keep track which GPU queries belong
    /// to this range.
    std::size_t mStartQueryIndex{};
    std::size_t mEndQueryIndex{};
  };

  /// A ScopedTimer is responsible for measuring time for its entire existence. The timer will
  /// start measuring upon creation and stop measuring on deletion.
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
    int32_t mID;
  };

  /// Access the singleton instance.
  static FrameTimings& get();

  /// To enable or disable time measuring globally.
  Property<bool> pEnableMeasurements = false;

  /// Get the frame time in milliseconds. This is the maximum of CPU and GPU time, excluding
  /// any waiting for vertical synchronization. As this requires a GPU timer query, the result will
  /// be based on the last-but-one frame. See the documentation of getRanges() for more details.
  Property<double> pFrameTime = 0.0;

  FrameTimings(FrameTimings const& other) = delete;
  FrameTimings(FrameTimings&& other)      = delete;

  FrameTimings& operator=(FrameTimings const& other) = delete;
  FrameTimings& operator=(FrameTimings&& other) = delete;

  ~FrameTimings() = default;

  /// Starts the time measurement for the current frame. No need to call this manually; the
  /// application is responsible for this.
  void startFrame();

  /// Ends the time measurement for the current frame. No need to call this manually; the
  /// application is responsible for this.
  void endFrame();

  /// Starts a timer with the given name and mode. You can use this interface, however the
  /// ScopedTimer is often more easy to use. The returned ID will be >= 0 if the timing range
  /// was actually started and -1 if pEnableMeasurements is set to false.
  int32_t startRange(std::string name, QueryMode mode = QueryMode::eBoth);

  /// Stops the timing range with the given ID. You can use this interface, however the ScopedTimer
  /// is often more easy to use.
  void endRange(int32_t id);

  /// This will retrieve the recorded ranges from the last-but-one frame. This is to prevent any
  /// synchronization between CPU and GPU: In one frame timings are recorded and queries are
  /// dispatched, then we wait one full frame until we attempt to read the query results. Then, in
  /// the third frame the results are available via this method.
  /// This will always contain at least one range for the entire frame. If pEnableMeasurements is
  /// set to true, it will contain all timing ranges created by all ScopedTimers. It may contain
  /// incomplete data for the frames in which pEnableMeasurements was toggled.
  std::vector<Range> const& getRanges();

 private:
  /// You should not need to instantiate this class. One singleton instance can be created with the
  /// static get() method above.
  FrameTimings();

  /// We have a triple-buffer of TimerPools.
  std::array<std::unique_ptr<TimerPool>, 3> mTimerPools;
  int32_t                                   mCurrentPool{};
  int32_t                                   mFullFrameTimingID{};
};

/// The TimerPool is used in a triple-buffer fashion internally by the FrameTimings class. You
/// will not need to use this class directly.
class CS_UTILS_EXPORT TimerPool {
 public:
  /// The TimerPool will allocate queryAllocationBucketSize GPU timer query objects initially.
  /// Whenever this amount is exhausted, a new batch of this size will be allocated.
  TimerPool(std::size_t queryAllocationBucketSize);

  /// Do not try to copy this class!
  TimerPool(TimerPool const&) = delete;
  void operator=(TimerPool const&) = delete;

  TimerPool(TimerPool&&) = delete;
  void operator=(TimerPool&&) = delete;

  ~TimerPool();

  /// Clears all recorded timing ranges.
  void reset();

  /// Starts a new timing range. The returned integer will always be >= 0 and can be used to end the
  /// range with the method below.
  int32_t startRange(std::string name, FrameTimings::QueryMode mode);

  /// Ends a previously started timing range. This will do nothing if the given id is invalid.
  void endRange(int32_t id);

  /// Fetches timestamps from GPU. This needs to be called before getRanges() and blocks until all
  /// queries are done.
  void fetchQueries();

  /// Returns the currently recorded ranges since the last call to reset(). It will not contain
  /// valid GPU timings until fetchQueries() was called.
  std::vector<FrameTimings::Range> const& getRanges() const;

 private:
  std::size_t startTimerQuery();

  std::size_t                      mQueryAllocationBucketSize{};
  std::vector<uint32_t>            mQueries;
  std::size_t                      mNextQueryID{};
  std::vector<FrameTimings::Range> mRanges;
  uint32_t                         mCurrentNestingLevel{};
};

} // namespace cs::utils

#endif // CS_UTILS_FRAME_TIMINGS_HPP
