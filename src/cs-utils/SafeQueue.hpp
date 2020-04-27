////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_THREADPOOL_HPP
#define CS_UTILS_THREADPOOL_HPP

#include <mutex>
#include <queue>

namespace cs::utils {

/// A very simple implementation of a thread-safe queue. You should only use this with a single
/// consumer.
template <typename T>
class SafeQueue {

 public:
  void push(T const& data) {
    std::lock_guard lock(mMutex);
    mQueue.push(data);
  }

  void push(T&& data) {
    std::lock_guard lock(mMutex);
    mQueue.emplace(data);
  }

  bool empty() const {
    std::lock_guard lock(mMutex);
    return mQueue.empty();
  }

  T& front() {
    std::lock_guard lock(mMutex);
    return mQueue.front();
  }

  T const& front() const {
    std::lock_guard lock(mMutex);
    return mQueue.front();
  }

  void pop() {
    std::lock_guard lock(mMutex);
    mQueue.pop();
  }

 private:
  std::queue<T>      mQueue;
  mutable std::mutex mMutex;
};

} // namespace cs::utils

#endif // CS_UTILS_THREADPOOL_HPP