////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ThreadPool.hpp"

#include <iostream>

namespace cs::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

ThreadPool::ThreadPool(size_t threads)
    : mStop(false) {
  for (size_t i = 0; i < threads; ++i) {
    mWorkers.emplace_back([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(mMutex);

          mCondition.wait(lock, [this] { return mStop || !mTasks.empty(); });

          if (mStop && mTasks.empty()) {
            return;
          }

          task = std::move(mTasks.top());
          mTasks.pop();
          ++mRunningTasks;
        }

        task();

        std::unique_lock<std::mutex> lock(mMutex);
        --mRunningTasks;
      }
    });
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(mMutex);
    mStop = true;
  }

  mCondition.notify_all();

  for (std::thread& worker : mWorkers) {
    worker.join();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils
