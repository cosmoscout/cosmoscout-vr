////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "FrameStats.hpp"

#include "logger.hpp"

#include <GL/glew.h>
#include <thread>
#include <utility>

namespace cs::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameStats::ScopedTimer::ScopedTimer(std::string name, TimerMode mode)
    : mID(FrameStats::get().startTimerQuery(std::move(name), mode)) {
}

FrameStats::ScopedTimer::~ScopedTimer() {
  FrameStats::get().endTimerQuery(mID);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameStats::ScopedSamplesCounter::ScopedSamplesCounter(std::string name)
    : mID(FrameStats::get().startSamplesQuery(std::move(name))) {
}

FrameStats::ScopedSamplesCounter::~ScopedSamplesCounter() {
  FrameStats::get().endSamplesQuery(mID);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameStats::ScopedPrimitivesCounter::ScopedPrimitivesCounter(std::string name)
    : mID(FrameStats::get().startPrimitivesQuery(std::move(name))) {
}

FrameStats::ScopedPrimitivesCounter::~ScopedPrimitivesCounter() {
  FrameStats::get().endPrimitivesQuery(mID);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameStats& FrameStats::get() {
  static FrameStats instance;
  return instance;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameStats::FrameStats() {
  pEnableMeasurements.connectAndTouch([this](bool enable) {
    for (auto& pool : mQueryPools) {
      if (enable) {
        pool = std::make_unique<QueryPool>(512);
      } else {
        // If measurements are disabled, we need at most two query objects, one for the start of the
        // full frame timing and one for the end of the full frame timing.
        pool = std::make_unique<QueryPool>(2);
      }
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameStats::startFrame() {

  // Advance the timer pool triple-buffer by one.
  mCurrentQueryPool = (mCurrentQueryPool + 1) % mQueryPools.size();

  // Fetch the query results for the oldest pool in our triple buffer. From this the getRanges()
  // call will read this frame.
  auto oldestPool = (mCurrentQueryPool + 1) % mQueryPools.size();
  mQueryPools.at(oldestPool)->fetchQueries();

  // Retrieve the pFrameTime from the oldest pool as well.
  double const toMilliSeconds = 0.000001;
  auto const&  timerResults   = mQueryPools.at(oldestPool)->getTimerQueryResults();

  if (!timerResults.empty()) {
    auto gpuTime = (timerResults[0].mGPUEnd - timerResults[0].mGPUStart) * toMilliSeconds;
    auto cpuTime = (timerResults[0].mCPUEnd - timerResults[0].mCPUStart) * toMilliSeconds;
    pFrameTime   = std::max(gpuTime, cpuTime);
  }

  // Reset the new current pool.
  auto const& pool = mQueryPools.at(mCurrentQueryPool);
  pool->reset();

  // Start the "root" full frame timing. This is always done, even if pEnableMeasurements is set to
  // false. This is required to get data for the pFrameTime property.
  mFullFrameTimingID = pool->startTimerQuery("Process Frame", FrameStats::TimerMode::eBoth);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameStats::endFrame() {

  // End the "root" full frame timing. This is always done, even if pEnableMeasurements is set to
  // false. This is required to get data for the pFrameTime property.
  mQueryPools.at(mCurrentQueryPool)->endTimerQuery(mFullFrameTimingID);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t FrameStats::startTimerQuery(std::string name, FrameStats::TimerMode mode) {

  // Only attempt to start the timing if pEnableMeasurements is set to true.
  if (pEnableMeasurements.get()) {
    return mQueryPools.at(mCurrentQueryPool)->startTimerQuery(std::move(name), mode);
  }

  return -1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t FrameStats::startSamplesQuery(std::string name) {

  // Only attempt to start the counting if pEnableMeasurements is set to true.
  if (pEnableMeasurements.get()) {
    return mQueryPools.at(mCurrentQueryPool)->startSamplesQuery(std::move(name));
  }

  return -1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t FrameStats::startPrimitivesQuery(std::string name) {

  // Only attempt to start the counting if pEnableMeasurements is set to true.
  if (pEnableMeasurements.get()) {
    return mQueryPools.at(mCurrentQueryPool)->startPrimitivesQuery(std::move(name));
  }

  return -1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameStats::endTimerQuery(int32_t id) {

  // Only attempt to end the timing if pEnableMeasurements is set to true.
  if (pEnableMeasurements.get()) {
    mQueryPools.at(mCurrentQueryPool)->endTimerQuery(id);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameStats::endSamplesQuery(int32_t id) {

  // Only attempt to end the counting if pEnableMeasurements is set to true.
  if (pEnableMeasurements.get()) {
    mQueryPools.at(mCurrentQueryPool)->endSamplesQuery(id);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameStats::endPrimitivesQuery(int32_t id) {

  // Only attempt to end the counting if pEnableMeasurements is set to true.
  if (pEnableMeasurements.get()) {
    mQueryPools.at(mCurrentQueryPool)->endPrimitivesQuery(id);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<FrameStats::TimerQueryResult> const& FrameStats::getTimerQueryResults() {

  // We return the ranges from the last-but-one frame.
  auto oldestPool = (mCurrentQueryPool + 1) % mQueryPools.size();
  return mQueryPools.at(oldestPool)->getTimerQueryResults();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<FrameStats::CounterQueryResult> const& FrameStats::getSamplesQueryResults() {

  // We return the ranges from the last-but-one frame.
  auto oldestPool = (mCurrentQueryPool + 1) % mQueryPools.size();
  return mQueryPools.at(oldestPool)->getSamplesQueryResults();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<FrameStats::CounterQueryResult> const& FrameStats::getPrimitivesQueryResults() {

  // We return the ranges from the last-but-one frame.
  auto oldestPool = (mCurrentQueryPool + 1) % mQueryPools.size();
  return mQueryPools.at(oldestPool)->getPrimitivesQueryResults();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

QueryPool::QueryPool(std::size_t queryAllocationBucketSize)
    : mQueryAllocationBucketSize(queryAllocationBucketSize) {

  mTimerQueries.mQueries.resize(mQueryAllocationBucketSize);
  mSamplesQueries.mQueries.resize(mQueryAllocationBucketSize);
  mPrimitivesQueries.mQueries.resize(mQueryAllocationBucketSize);

  glGenQueries(static_cast<GLsizei>(queryAllocationBucketSize), mTimerQueries.mQueries.data());
  glGenQueries(static_cast<GLsizei>(queryAllocationBucketSize), mSamplesQueries.mQueries.data());
  glGenQueries(static_cast<GLsizei>(queryAllocationBucketSize), mPrimitivesQueries.mQueries.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

QueryPool::~QueryPool() {
  glDeleteQueries(
      static_cast<GLsizei>(mTimerQueries.mQueries.size()), mTimerQueries.mQueries.data());
  glDeleteQueries(
      static_cast<GLsizei>(mSamplesQueries.mQueries.size()), mSamplesQueries.mQueries.data());
  glDeleteQueries(
      static_cast<GLsizei>(mPrimitivesQueries.mQueries.size()), mPrimitivesQueries.mQueries.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void QueryPool::reset() {
  mTimerQueryResults.clear();
  mSamplesQueryResults.clear();
  mPrimitivesQueryResults.clear();

  mTimerQueries.mNextID      = 0;
  mSamplesQueries.mNextID    = 0;
  mPrimitivesQueries.mNextID = 0;

  mCurrentNestingLevel = 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t QueryPool::startTimerQuery(std::string name, FrameStats::TimerMode mode) {

  FrameStats::TimerQueryResult result;
  result.mMode         = mode;
  result.mName         = std::move(name);
  result.mNestingLevel = mCurrentNestingLevel++;

  // Start the GPU result if necessary.
  if (mode == FrameStats::TimerMode::eGPU || mode == FrameStats::TimerMode::eBoth) {
    result.mStartQueryIndex = startTimerQuery();
  }

  // Start the CPU result if necessary.
  if (mode == FrameStats::TimerMode::eCPU || mode == FrameStats::TimerMode::eBoth) {
    result.mCPUStart = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }

  mTimerQueryResults.push_back(std::move(result));

  // Return the index at which this result was inserted.
  return static_cast<int32_t>(mTimerQueryResults.size() - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t QueryPool::startSamplesQuery(std::string name) {
  FrameStats::CounterQueryResult result;
  result.mName       = std::move(name);
  result.mQueryIndex = startSamplesQuery();

  mSamplesQueryResults.push_back(std::move(result));

  // Return the index at which this result was inserted.
  return static_cast<int32_t>(mSamplesQueryResults.size() - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t QueryPool::startPrimitivesQuery(std::string name) {
  FrameStats::CounterQueryResult result;
  result.mName       = std::move(name);
  result.mQueryIndex = startPrimitivesQuery();

  mPrimitivesQueryResults.push_back(std::move(result));

  // Return the index at which this result was inserted.
  return static_cast<int32_t>(mPrimitivesQueryResults.size() - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void QueryPool::endTimerQuery(int32_t id) {

  // Abort if no valid ID was given.
  if (id < 0 || id >= static_cast<int32_t>(mTimerQueryResults.size())) {
    return;
  }

  // End the GPU range if necessary.
  if (mTimerQueryResults[id].mMode == FrameStats::TimerMode::eGPU ||
      mTimerQueryResults[id].mMode == FrameStats::TimerMode::eBoth) {
    mTimerQueryResults[id].mEndQueryIndex = startTimerQuery();
  }

  // End the CPU range if necessary.
  if (mTimerQueryResults[id].mMode == FrameStats::TimerMode::eCPU ||
      mTimerQueryResults[id].mMode == FrameStats::TimerMode::eBoth) {
    mTimerQueryResults[id].mCPUEnd =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }

  --mCurrentNestingLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void QueryPool::endSamplesQuery(int32_t id) {

  // Abort if no valid ID was given.
  if (id < 0 || id >= static_cast<int32_t>(mSamplesQueryResults.size())) {
    return;
  }

  glEndQuery(GL_SAMPLES_PASSED);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void QueryPool::endPrimitivesQuery(int32_t id) {

  // Abort if no valid ID was given.
  if (id < 0 || id >= static_cast<int32_t>(mPrimitivesQueryResults.size())) {
    return;
  }

  glEndQuery(GL_PRIMITIVES_GENERATED);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void QueryPool::fetchQueries() {

  // Wait for the last query to finish.
  waitForQueries(mTimerQueries);
  waitForQueries(mSamplesQueries);
  waitForQueries(mPrimitivesQueries);

  // Get the query results.
  std::vector<uint64_t> timerQueryResults      = getQueryResults(mTimerQueries);
  std::vector<uint64_t> samplesQueryResults    = getQueryResults(mSamplesQueries);
  std::vector<uint64_t> primitivesQueryResults = getQueryResults(mPrimitivesQueries);

  for (std::size_t i = 0; i < mTimerQueryResults.size(); ++i) {
    if (mTimerQueryResults[i].mMode == FrameStats::TimerMode::eGPU ||
        mTimerQueryResults[i].mMode == FrameStats::TimerMode::eBoth) {
      mTimerQueryResults[i].mGPUStart = timerQueryResults[mTimerQueryResults[i].mStartQueryIndex];
      mTimerQueryResults[i].mGPUEnd   = timerQueryResults[mTimerQueryResults[i].mEndQueryIndex];
    }
  }

  for (std::size_t i = 0; i < mSamplesQueryResults.size(); ++i) {
    mSamplesQueryResults[i].mCount = samplesQueryResults[mSamplesQueryResults[i].mQueryIndex];
  }

  for (std::size_t i = 0; i < mPrimitivesQueryResults.size(); ++i) {
    mPrimitivesQueryResults[i].mCount =
        primitivesQueryResults[mPrimitivesQueryResults[i].mQueryIndex];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<FrameStats::TimerQueryResult> const& QueryPool::getTimerQueryResults() const {
  return mTimerQueryResults;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<FrameStats::CounterQueryResult> const& QueryPool::getSamplesQueryResults() const {
  return mSamplesQueryResults;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<FrameStats::CounterQueryResult> const& QueryPool::getPrimitivesQueryResults() const {
  return mPrimitivesQueryResults;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::size_t QueryPool::startTimerQuery() {
  if (mTimerQueries.mNextID >= mTimerQueries.mQueries.size()) {
    auto currentSize = mTimerQueries.mQueries.size();
    mTimerQueries.mQueries.resize(currentSize + mQueryAllocationBucketSize, 0);
    glGenQueries(static_cast<GLsizei>(mQueryAllocationBucketSize),
        mTimerQueries.mQueries.data() + currentSize);
    logger().info("reallocating startTimerQuery");
  }

  glQueryCounter(mTimerQueries.mQueries[mTimerQueries.mNextID], GL_TIMESTAMP);

  return mTimerQueries.mNextID++;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::size_t QueryPool::startSamplesQuery() {
  if (mSamplesQueries.mNextID >= mSamplesQueries.mQueries.size()) {
    auto currentSize = mSamplesQueries.mQueries.size();
    mSamplesQueries.mQueries.resize(currentSize + mQueryAllocationBucketSize, 0);
    glGenQueries(static_cast<GLsizei>(mQueryAllocationBucketSize),
        mSamplesQueries.mQueries.data() + currentSize);
    logger().info("reallocating startSamplesQuery");
  }

  glBeginQuery(GL_SAMPLES_PASSED, mSamplesQueries.mQueries[mSamplesQueries.mNextID]);

  return mSamplesQueries.mNextID++;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::size_t QueryPool::startPrimitivesQuery() {
  if (mPrimitivesQueries.mNextID >= mPrimitivesQueries.mQueries.size()) {
    auto currentSize = mPrimitivesQueries.mQueries.size();
    mPrimitivesQueries.mQueries.resize(currentSize + mQueryAllocationBucketSize, 0);
    glGenQueries(static_cast<GLsizei>(mQueryAllocationBucketSize),
        mPrimitivesQueries.mQueries.data() + currentSize);
    logger().info("reallocating startPrimitivesQuery");
  }

  glBeginQuery(GL_PRIMITIVES_GENERATED, mPrimitivesQueries.mQueries[mPrimitivesQueries.mNextID]);

  return mPrimitivesQueries.mNextID++;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void QueryPool::waitForQueries(Queries const& queries) const {
  int32_t queriesDone = 0;
  while (queries.mNextID > 0 && !queriesDone) {
    glGetQueryObjectiv(
        queries.mQueries[queries.mNextID - 1], GL_QUERY_RESULT_AVAILABLE, &queriesDone);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<uint64_t> QueryPool::getQueryResults(Queries const& queries) const {
  std::vector<uint64_t> results(queries.mNextID);
  for (std::size_t i = 0; i < queries.mNextID; ++i) {
    glGetQueryObjectui64v(queries.mQueries[i], GL_QUERY_RESULT, &results[i]);
  }

  return results;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils
