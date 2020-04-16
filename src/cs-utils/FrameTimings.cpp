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

namespace {
int                                            s_iCurrentInstance = 0;
std::array<std::shared_ptr<TimerQueryPool>, 2> s_pTimerQueryPoolInstances{};
std::string s_sLastRangeKey{}; // NOLINT(fuchsia-statically-constructed-objects)
} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameTimings::ScopedTimer::ScopedTimer(std::string name, QueryMode mode)
    : mName(std::move(name)) {
  FrameTimings::start(mName, mode);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameTimings::ScopedTimer::~ScopedTimer() {
  try {
    FrameTimings::end(mName);
  } catch (...) {}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameTimings::start(std::string const& name, QueryMode mode) {
  if (s_pTimerQueryPoolInstances.at(s_iCurrentInstance)) {
    s_pTimerQueryPoolInstances.at(s_iCurrentInstance)->start(name, mode);
    s_sLastRangeKey = name;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameTimings::end(std::string const& name) {
  if (s_pTimerQueryPoolInstances.at(s_iCurrentInstance)) {
    s_pTimerQueryPoolInstances.at(s_iCurrentInstance)->end(name);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameTimings::end() {
  if (s_pTimerQueryPoolInstances.at(s_iCurrentInstance)) {
    s_pTimerQueryPoolInstances.at(s_iCurrentInstance)->end(s_sLastRangeKey);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unordered_map<std::string, FrameTimings::QueryResult>
FrameTimings::getCalculatedQueryResults() const {
  std::unordered_map<std::string, QueryResult> result;

  if (!s_pTimerQueryPoolInstances.at(s_iCurrentInstance)) {
    return result;
  }

  if (pEnableMeasurements.get()) {
    for (auto const& ranges :
        s_pTimerQueryPoolInstances.at(s_iCurrentInstance)->getQueryResults()) {
      uint64_t timeGPU(0);
      uint64_t timeCPU(0);
      for (auto const& range : ranges.second) {
        timeGPU += range.mGPUTime;
        timeCPU += range.mCPUTime;
      }

      result[ranges.first] = {timeGPU, timeCPU};
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FrameTimings::FrameTimings() {
  std::size_t const maxNRofTimings = 512;
  pEnableMeasurements.connect([maxNRofTimings](bool enable) {
    if (enable) {
      s_pTimerQueryPoolInstances[0] = std::make_shared<TimerQueryPool>(maxNRofTimings);
      s_pTimerQueryPoolInstances[1] = std::make_shared<TimerQueryPool>(maxNRofTimings);
    } else {
      s_pTimerQueryPoolInstances[0] = nullptr;
      s_pTimerQueryPoolInstances[1] = nullptr;
    }
  });
  mFullFrameTimerPools[0] = std::make_shared<TimerQueryPool>(2);
  mFullFrameTimerPools[1] = std::make_shared<TimerQueryPool>(2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameTimings::startFullFrameTiming() {
  mFullFrameTimerPools.at(mCurrentIndex)->start("FullFrame", FrameTimings::QueryMode::eBoth);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameTimings::endFullFrameTiming() {
  mFullFrameTimerPools.at(mCurrentIndex)->end("FullFrame");
  mCurrentIndex = (mCurrentIndex + 1) % 2;
  mFullFrameTimerPools.at(mCurrentIndex)->calculateQueryResults();

  double const epsilon      = 0.000001;
  auto         queryResults = mFullFrameTimerPools.at(mCurrentIndex)->getQueryResults();
  if (!queryResults.empty() && !queryResults.begin()->second.empty()) {
    pFrameTime = std::max(queryResults.begin()->second[0].mGPUTime * epsilon,
        queryResults.begin()->second[0].mCPUTime * epsilon);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FrameTimings::update() const {
  s_iCurrentInstance = (s_iCurrentInstance + 1) % 2;

  if (!s_pTimerQueryPoolInstances.at(s_iCurrentInstance)) {
    return;
  }

  if (pEnableMeasurements.get()) {
    s_pTimerQueryPoolInstances.at(s_iCurrentInstance)->calculateQueryResults();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimerQueryPool::TimerQueryPool(std::size_t max_size)
    : mMaxSize(max_size)
    , mQueries(max_size, 0)
    , mTimestamps(max_size, 0)
    , mQueryDone(0)
    , mIndex(0) {
  glGenQueries(static_cast<GLsizei>(max_size), mQueries.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimerQueryPool::start(std::string const& name, FrameTimings::QueryMode mode) {
  QueryRange range;
  range.mMode = mode;

  if (mode == FrameTimings::QueryMode::eGPU || mode == FrameTimings::QueryMode::eBoth) {
    auto t = timestamp();
    if (t) {
      range.mGPUStart = *t;
    } else {
      logger()->warn(
          "Failed to start timer query: No more Timestamps available (mMaxSize={}, mIndex={})!",
          mMaxSize, mIndex);
    }
  }

  if (mode == FrameTimings::QueryMode::eCPU || mode == FrameTimings::QueryMode::eBoth) {
    auto now        = std::chrono::high_resolution_clock::now();
    range.mCPUStart = now;
  }

  mQueryRanges[name].push_back(range);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimerQueryPool::end(std::string const& name) {
  if (mQueryRanges.empty()) {
    return;
  }

  auto iter = mQueryRanges.find(name);
  if (iter == mQueryRanges.end()) {
    logger()->warn("Failed to end timer query: Unknown key '{}'!", name);
    return;
  }

  auto& range = iter->second.back();
  if (range.mMode == FrameTimings::QueryMode::eGPU ||
      range.mMode == FrameTimings::QueryMode::eBoth) {
    auto t = timestamp();
    if (t) {
      range.mGPUEnd = *t;
    } else {
      logger()->warn("Failed to end timer query: No more Timestamps available!");
    }
  }

  if (range.mMode == FrameTimings::QueryMode::eCPU ||
      range.mMode == FrameTimings::QueryMode::eBoth) {
    range.mCPUEnd = std::chrono::high_resolution_clock::now();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::size_t> TimerQueryPool::timestamp() {
  if (mIndex < mMaxSize) {
    glQueryCounter(mQueries[mIndex], GL_TIMESTAMP);
    std::size_t inserted_at = mIndex;
    ++mIndex;
    return inserted_at;
  }
  return std::nullopt;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimerQueryPool::calculateQueryResults() {
  // Fetch timestamps from gpu and calculate time diffs.
  if (mIndex == 0) {
    return;
  }

  mQueryDone = 0;

  // Wait for queries to finish.
  while (!mQueryDone) {
    glGetQueryObjectiv(mQueries[mIndex - 1], GL_QUERY_RESULT_AVAILABLE, &mQueryDone);
  }

  // Get the query results.
  for (std::size_t i = 0; i < mIndex; ++i) {
    glGetQueryObjectui64v(mQueries[i], GL_QUERY_RESULT, &mTimestamps[i]);
  }

  mQueryResults.clear();

  for (auto& pair : mQueryRanges) {
    for (auto& range : pair.second) {
      FrameTimings::QueryResult result;
      result.mGPUTime = mTimestamps[range.mGPUEnd] - mTimestamps[range.mGPUStart];
      result.mCPUTime = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(range.mCPUEnd - range.mCPUStart)
              .count());
      mQueryResults[pair.first].push_back(result);
    }
  }

  // Reset index.
  mIndex = 0;
  mQueryRanges.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unordered_map<std::string, std::vector<FrameTimings::QueryResult>> const&
TimerQueryPool::getQueryResults() const {
  return mQueryResults;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils
