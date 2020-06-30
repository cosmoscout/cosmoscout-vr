////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ScopedTimer.hpp"

#include "../logger.hpp"

#include <chrono>
#include <utility>

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

ScopedTimer::ScopedTimer(std::string name)
    : mName(std::move(name))
    , mStartTime(GetNow()) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ScopedTimer::~ScopedTimer() {
  double now = GetNow();
  logger().info("{}: {} ms", mName, now - mStartTime);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double ScopedTimer::GetNow() {
  auto         time        = std::chrono::system_clock::now();
  auto         since_epoch = time.time_since_epoch();
  double const microToNano = 0.001;
  return std::chrono::duration_cast<std::chrono::microseconds>(since_epoch).count() * microToNano;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
