////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_UTILS_FRAME_STATS_HPP
#define CS_UTILS_FRAME_STATS_HPP

#include "Property.hpp"
#include "cs_utils_export.hpp"

#include <array>
#include <chrono>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace cs::utils {

class QueryPool;

/// This is a singleton class which can be used in the main thread to record CPU und GPU timings as
/// well as generated fragments and primitives. For the timing, this class actively supports
/// measuring nested ranges so it's perfectively fine to start ranges for large parts of the code
/// and multiple smaller subranges for smaller parts. To measure the time spent on the CPU or the
/// GPU by a specific block of code, simply create a FrameStats::ScopedTimer. This will start a
/// measuring range in its constructor and and end the range in its destructor.
/// The ScopedCounter does not support nesting, so you have to ensure that you do not start to
/// ScopedCounters at the same time.
class CS_UTILS_EXPORT FrameStats {
 public:
  /// Defines which timings should be measured.
  enum class TimerMode {
    eCPU, ///< Only the CPU time will be measured.
    eGPU, ///< Only the GPU time will be measured.
    eBoth
  };

  /// This struct contains information on one specific timing range. It is used internally by the
  /// FrameStats singleton and can be accessed via its getTimerQueryResults() method.
  struct TimerQueryResult {

    /// The name of the range as it was passed to the constructor of the ScopedTimer or the
    /// FrameStats::startRange() method.
    std::string mName;

    /// This contains the number of timing ranges which were active when this range was started.
    uint32_t mNestingLevel{};

    /// Timestamps in nanoseconds when the range started / ended on the CPU / GPU. They will be
    /// filled with data only if the TimerMode was set accordingly.
    int64_t mCPUStart{};
    int64_t mCPUEnd{};
    int64_t mGPUStart{};
    int64_t mGPUEnd{};

    /// The mode this range was started with.
    TimerMode mMode;

    /// These are used internally by the FrameStats class to keep track which GPU queries belong
    /// to this range.
    std::size_t mStartQueryIndex{};
    std::size_t mEndQueryIndex{};
  };

  /// This struct contains information on one specific counting range. It is used internally by the
  /// FrameStats singleton and can be accessed via its getCounterQueryResults() method.
  struct CounterQueryResult {
    std::string mName;
    int64_t     mCount{};
    std::size_t mQueryIndex{};
  };

  /// A ScopedCounter is responsible for counting generated fragments and primitives during its
  /// entire existence. The counter will start measuring upon creation and stop measuring on
  /// deletion.
  class CS_UTILS_EXPORT ScopedTimer {
   public:
    /// @param name The name of the counter.
    /// @param mode The mode of querying. See CounterMode for more info.
    explicit ScopedTimer(std::string name, TimerMode mode = TimerMode::eBoth);

    ScopedTimer(ScopedTimer const& other) = delete;
    ScopedTimer(ScopedTimer&& other)      = delete;

    ScopedTimer& operator=(ScopedTimer const& other) = delete;
    ScopedTimer& operator=(ScopedTimer&& other) = delete;

    ~ScopedTimer();

   private:
    int32_t mID;
  };

  class CS_UTILS_EXPORT ScopedSamplesCounter {
   public:
    /// @param name The name of the counter.
    explicit ScopedSamplesCounter(std::string name);

    ScopedSamplesCounter(ScopedSamplesCounter const& other) = delete;
    ScopedSamplesCounter(ScopedSamplesCounter&& other)      = delete;

    ScopedSamplesCounter& operator=(ScopedSamplesCounter const& other) = delete;
    ScopedSamplesCounter& operator=(ScopedSamplesCounter&& other) = delete;

    ~ScopedSamplesCounter();

   private:
    int32_t mID;
  };

  class CS_UTILS_EXPORT ScopedPrimitivesCounter {
   public:
    /// @param name The name of the counter.
    explicit ScopedPrimitivesCounter(std::string name);

    ScopedPrimitivesCounter(ScopedPrimitivesCounter const& other) = delete;
    ScopedPrimitivesCounter(ScopedPrimitivesCounter&& other)      = delete;

    ScopedPrimitivesCounter& operator=(ScopedPrimitivesCounter const& other) = delete;
    ScopedPrimitivesCounter& operator=(ScopedPrimitivesCounter&& other) = delete;

    ~ScopedPrimitivesCounter();

   private:
    int32_t mID;
  };

  /// Access the singleton instance.
  static FrameStats& get();

  /// To enable or disable time measuring globally.
  Property<bool> pEnableMeasurements = false;

  /// Get the frame time in milliseconds. This is the maximum of CPU and GPU time, excluding
  /// any waiting for vertical synchronization. As this requires a GPU timer query, the result will
  /// be based on the last-but-one frame. See the documentation of getRanges() for more details.
  Property<double> pFrameTime = 0.0;

  FrameStats(FrameStats const& other) = delete;
  FrameStats(FrameStats&& other)      = delete;

  FrameStats& operator=(FrameStats const& other) = delete;
  FrameStats& operator=(FrameStats&& other) = delete;

  ~FrameStats() = default;

  /// Starts the time measurement for the current frame. No need to call this manually; the
  /// application is responsible for this.
  void startFrame();

  /// Ends the time measurement for the current frame. No need to call this manually; the
  /// application is responsible for this.
  void endFrame();

  /// Starts a timer / counter with the given name and mode. You can use this interface, however the
  /// ScopedTimer and ScopedCounter are often more easy to use. The returned ID will be >= 0 if the
  /// timing range was actually started and -1 if pEnableMeasurements is set to false or a counter
  /// of the given type is already active (counter ranges cannot be nested).
  int32_t startTimerQuery(std::string name, TimerMode mode = TimerMode::eBoth);
  int32_t startSamplesQuery(std::string name);
  int32_t startPrimitivesQuery(std::string name);

  /// Stops the timing range with the given ID. You can use this interface, however the ScopedTimer
  /// is often more easy to use.
  void endTimerQuery(int32_t id);
  void endSamplesQuery(int32_t id);
  void endPrimitivesQuery(int32_t id);

  /// This will retrieve the recorded ranges from the last-but-one frame. This is to prevent any
  /// synchronization between CPU and GPU: In one frame timings are recorded and queries are
  /// dispatched, then we wait one full frame until we attempt to read the query results. Then, in
  /// the third frame the results are available via this method.
  /// This will always contain at least one range for the entire frame. If pEnableMeasurements is
  /// set to true, it will contain all timing ranges created by all ScopedTimers. It may contain
  /// incomplete data for the frames in which pEnableMeasurements was toggled.
  std::vector<TimerQueryResult> const&   getTimerQueryResults();
  std::vector<CounterQueryResult> const& getSamplesQueryResults();
  std::vector<CounterQueryResult> const& getPrimitivesQueryResults();

 private:
  /// You should not need to instantiate this class. One singleton instance can be created with the
  /// static get() method above.
  FrameStats();

  /// We have a triple-buffer of QueryPools.
  std::array<std::unique_ptr<QueryPool>, 3> mQueryPools;
  int32_t                                   mCurrentQueryPool{};

  int32_t mFullFrameTimingID{};
};

/// The QueryPool is used in a triple-buffer fashion internally by the FrameStats class. You
/// will not need to use this class directly.
class CS_UTILS_EXPORT QueryPool {
 public:
  /// The QueryPool will allocate queryAllocationBucketSize GPU timer query objects initially.
  /// Whenever this amount is exhausted, a new batch of this size will be allocated.
  QueryPool(std::size_t queryAllocationBucketSize);

  /// Do not try to copy this class!
  QueryPool(QueryPool const&) = delete;
  void operator=(QueryPool const&) = delete;

  QueryPool(QueryPool&&) = delete;
  void operator=(QueryPool&&) = delete;

  ~QueryPool();

  /// Clears all recorded timing ranges.
  void reset();

  /// Starts a new timing range. The returned integer will always be >= 0 and can be used to end the
  /// range with the method below.
  int32_t startTimerQuery(std::string name, FrameStats::TimerMode mode);
  int32_t startSamplesQuery(std::string name);
  int32_t startPrimitivesQuery(std::string name);

  /// Ends a previously started timing range. This will do nothing if the given id is invalid.
  void endTimerQuery(int32_t id);
  void endSamplesQuery(int32_t id);
  void endPrimitivesQuery(int32_t id);

  /// Fetches timestamps from GPU. This needs to be called before getRanges() and blocks until all
  /// queries are done.
  void fetchQueries();

  /// Returns the currently recorded ranges since the last call to reset(). It will not contain
  /// valid GPU timings until fetchQueries() was called.
  std::vector<FrameStats::TimerQueryResult> const&   getTimerQueryResults() const;
  std::vector<FrameStats::CounterQueryResult> const& getSamplesQueryResults() const;
  std::vector<FrameStats::CounterQueryResult> const& getPrimitivesQueryResults() const;

 private:
  struct Queries {
    std::vector<uint32_t> mQueries;
    std::size_t           mNextID{};
  };

  std::size_t startTimerQuery();
  std::size_t startSamplesQuery();
  std::size_t startPrimitivesQuery();

  void                  waitForQueries(Queries const& queries) const;
  std::vector<uint64_t> getQueryResults(Queries const& queries) const;

  std::size_t mQueryAllocationBucketSize{};

  Queries mTimerQueries{};
  Queries mSamplesQueries{};
  Queries mPrimitivesQueries{};

  std::vector<FrameStats::TimerQueryResult>   mTimerQueryResults;
  std::vector<FrameStats::CounterQueryResult> mSamplesQueryResults;
  std::vector<FrameStats::CounterQueryResult> mPrimitivesQueryResults;

  uint32_t mCurrentNestingLevel{};
};

} // namespace cs::utils

#endif // CS_UTILS_FRAME_STATS_HPP
