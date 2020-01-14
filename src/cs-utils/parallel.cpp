////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "parallel.hpp"
#include <thread>

namespace cs::utils {

void parallelFor(size_t count, size_t numThreads, std::function<void(size_t)> const& f) {
  std::vector<std::thread> threads;

  size_t amountPerThread = count / numThreads + 1;
  for (size_t t = 0; t < numThreads; ++t) {
    threads.emplace_back([&, t] {
      size_t start = t * amountPerThread;
      size_t end   = start + amountPerThread;

      for (size_t i = start; i < end && i < count; ++i) {
        f(i);
      }
    });
  }

  for (auto&& thread : threads) {
    thread.join();
  }
}

void parallelFor(size_t count, std::function<void(size_t)> const& f) {
  parallelFor(count, std::thread::hardware_concurrency(), f);
}

}