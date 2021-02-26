////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FrameTimings.hpp"

#include "logger.hpp"

#include <GL/glew.h>
#include <thread>
#include <utility>

namespace cs::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameTimings::ScopedTimer::ScopedTimer(std::string name, QueryMode mode)
    : mID(FrameTimings::get().startRange(std::move(name), mode)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameTimings::ScopedTimer::~ScopedTimer() {
  FrameTimings::get().endRange(mID);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameTimings& FrameTimings::get() {
  static FrameTimings instance;
  return instance;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameTimings::FrameTimings() {
  pEnableMeasurements.connectAndTouch([this](bool enable) {
    for (auto& pool : mTimerPools) {
      if (enable) {
        pool = std::make_unique<TimerPool>(512);
      } else {
        // If measurements are disabled, we need at most two query objects, one for the start of the
        // full frame timing and one for the end of the full frame timing.
        pool = std::make_unique<TimerPool>(2);
      }
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameTimings::startFrame() {

  // Advance the timer pool triple-buffer by one.
  mCurrentPool = (mCurrentPool + 1) % mTimerPools.size();

  // Fetch the query results for the oldest pool in our triple buffer. From this the getRanges()
  // call will read this frame.
  auto oldestPool = (mCurrentPool + 1) % mTimerPools.size();
  mTimerPools.at(oldestPool)->fetchQueries();

  // Retrieve the pFrameTime from the oldest pool as well.
  double const toMilliSeconds = 0.000001;
  auto const&  ranges         = mTimerPools.at(oldestPool)->getRanges();

  if (!ranges.empty()) {
    auto gpuTime = (ranges[0].mGPUEnd - ranges[0].mGPUStart) * toMilliSeconds;
    auto cpuTime = (ranges[0].mCPUEnd - ranges[0].mCPUStart) * toMilliSeconds;
    pFrameTime   = std::max(gpuTime, cpuTime);
  }

  // Reset the new current pool.
  auto const& pool = mTimerPools.at(mCurrentPool);
  pool->reset();

  // Start the "root" full frame timing. This is always done, even if pEnableMeasurements is set to
  // false. This is required to get data for the pFrameTime property.
  mFullFrameTimingID = pool->startRange("Process Frame", FrameTimings::QueryMode::eBoth);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameTimings::endFrame() {

  // End the "root" full frame timing. This is always done, even if pEnableMeasurements is set to
  // false. This is required to get data for the pFrameTime property.
  mTimerPools.at(mCurrentPool)->endRange(mFullFrameTimingID);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t FrameTimings::startRange(std::string name, QueryMode mode) {

  // Only attempt to start the timing if pEnableMeasurements is set to true.
  if (pEnableMeasurements.get()) {
    return mTimerPools.at(mCurrentPool)->startRange(std::move(name), mode);
  }

  return -1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameTimings::endRange(int32_t id) {

  // Only attempt to end the timing if pEnableMeasurements is set to true.
  if (pEnableMeasurements.get()) {
    mTimerPools.at(mCurrentPool)->endRange(id);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<FrameTimings::Range> const& FrameTimings::getRanges() {

  // We return the ranges from the last-but-one frame.
  auto oldestPool = (mCurrentPool + 1) % mTimerPools.size();
  return mTimerPools.at(oldestPool)->getRanges();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimerPool::TimerPool(std::size_t queryAllocationBucketSize)
    : mQueryAllocationBucketSize(queryAllocationBucketSize)
    , mQueries(queryAllocationBucketSize, 0) {
  glGenQueries(static_cast<GLsizei>(queryAllocationBucketSize), mQueries.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimerPool::~TimerPool() {
  glDeleteQueries(static_cast<GLsizei>(mQueries.size()), mQueries.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimerPool::reset() {
  mNextQueryID         = 0;
  mCurrentNestingLevel = 0;
  mRanges.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t TimerPool::startRange(std::string name, FrameTimings::QueryMode mode) {

  FrameTimings::Range range;
  range.mMode         = mode;
  range.mName         = std::move(name);
  range.mNestingLevel = mCurrentNestingLevel++;

  // Start the GPU range if necessary.
  if (mode == FrameTimings::QueryMode::eGPU || mode == FrameTimings::QueryMode::eBoth) {
    range.mStartQueryIndex = startTimerQuery();
  }

  // Start the CPU range if necessary.
  if (mode == FrameTimings::QueryMode::eCPU || mode == FrameTimings::QueryMode::eBoth) {
    range.mCPUStart = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }

  mRanges.push_back(std::move(range));

  // Return the index at which this range was inserted.
  return static_cast<int32_t>(mRanges.size() - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimerPool::endRange(int32_t id) {

  // Abort if no valid ID was given.
  if (id < 0 || id >= static_cast<int32_t>(mRanges.size())) {
    return;
  }

  // End the GPU range if necessary.
  if (mRanges[id].mMode == FrameTimings::QueryMode::eGPU ||
      mRanges[id].mMode == FrameTimings::QueryMode::eBoth) {
    mRanges[id].mEndQueryIndex = startTimerQuery();
  }

  // End the CPU range if necessary.
  if (mRanges[id].mMode == FrameTimings::QueryMode::eCPU ||
      mRanges[id].mMode == FrameTimings::QueryMode::eBoth) {
    mRanges[id].mCPUEnd = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }

  --mCurrentNestingLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimerPool::fetchQueries() {
  // No query has been issued, so nothing to update.
  if (mNextQueryID == 0) {
    return;
  }

  // Wait for the last query to finish.
  int32_t queriesDone = 0;
  while (!queriesDone) {
    glGetQueryObjectiv(mQueries[mNextQueryID - 1], GL_QUERY_RESULT_AVAILABLE, &queriesDone);
  }

  // Get the query results.
  std::vector<uint64_t> queryResults(mNextQueryID);
  for (std::size_t i = 0; i < mNextQueryID; ++i) {
    glGetQueryObjectui64v(mQueries[i], GL_QUERY_RESULT, &queryResults[i]);
  }

  for (std::size_t i = 0; i < mRanges.size(); ++i) {
    if (mRanges[i].mMode == FrameTimings::QueryMode::eGPU ||
        mRanges[i].mMode == FrameTimings::QueryMode::eBoth) {
      mRanges[i].mGPUStart = queryResults[mRanges[i].mStartQueryIndex];
      mRanges[i].mGPUEnd   = queryResults[mRanges[i].mEndQueryIndex];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<FrameTimings::Range> const& TimerPool::getRanges() const {
  return mRanges;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::size_t TimerPool::startTimerQuery() {
  if (mNextQueryID >= mQueries.size()) {
    auto currentSize = mQueries.size();
    mQueries.resize(currentSize + mQueryAllocationBucketSize, 0);
    glGenQueries(static_cast<GLsizei>(mQueryAllocationBucketSize), mQueries.data() + currentSize);
  }

  glQueryCounter(mQueries[mNextQueryID], GL_TIMESTAMP);
  return mNextQueryID++;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils
