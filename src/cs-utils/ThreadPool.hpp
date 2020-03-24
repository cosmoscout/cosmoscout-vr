////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_THREADPOOL_HPP
#define CS_UTILS_THREADPOOL_HPP

#include "cs_utils_export.hpp"

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stack>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cs::utils {

/// This is based on https://github.com/progschj/ThreadPool
class CS_UTILS_EXPORT ThreadPool {
 public:
  /// Creates a new ThreadPool with the specified amount of threads.
  ThreadPool(size_t threadCount);
  virtual ~ThreadPool();

  /// Adds a new work item to the pool.
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(mMutex);

      if (mStop) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }

      mTasks.emplace([task]() { (*task)(); });
    }
    mCondition.notify_one();
    return res;
  }

  /// Returns the amount of tasks that await execution.
  uint32_t getPendingTaskCount() const {
    std::unique_lock<std::mutex> lock(mMutex);
    return static_cast<uint32_t>(mTasks.size());
  }

  /// Returns the number of tasks that currently are being executed.
  uint32_t getRunningTaskCount() const {
    std::unique_lock<std::mutex> lock(mMutex);
    return mRunningTasks;
  }

  /// Retruns true when there are no more tasks running or pending.
  bool hasFinished() const {
    return getPendingTaskCount() + getRunningTaskCount() == 0;
  }

 private:
  std::vector<std::thread>          mWorkers;
  std::stack<std::function<void()>> mTasks;
  mutable std::mutex                mMutex;
  std::condition_variable           mCondition;
  bool                              mStop;
  uint32_t                          mRunningTasks = 0;
};

} // namespace cs::utils

#endif // CS_UTILS_THREADPOOL_HPP
