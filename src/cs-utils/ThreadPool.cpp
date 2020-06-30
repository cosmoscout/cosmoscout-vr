////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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
