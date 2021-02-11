////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_SIGNAL_HPP
#define CS_UTILS_SIGNAL_HPP

#include <functional>
#include <iostream>
#include <map>

#include <cs_utils_export.hpp>
#include <spdlog/spdlog.h>

namespace cs::utils {

CS_UTILS_EXPORT spdlog::logger& logger();

/// A signal object may call multiple slots with the same signature. You can connect functions to
/// the signal which will be called when the emit() method on the signal object is invoked. Any
/// argument passed to emit() will be passed to the given functions.
template <typename... Args>
class Signal {

 public:
  Signal() = default;

  /// Copy creates new signal.
  Signal(Signal const& /*unused*/) {
  }

  Signal(Signal&& other) noexcept
      : mSlots(std::move(other.mSlots))
      , mCurrentID(other.mCurrentID) {
  }

  Signal& operator=(Signal&& other) noexcept {
    if (this != &other) {
      mSlots     = std::move(other.mSlots);
      mCurrentID = other.mCurrentID;
    }

    return *this;
  }

  ~Signal() = default;

  /// Connects a std::function to the signal. The returned value can be used to disconnect the
  /// function again.
  int connect(std::function<void(Args...)> const& slot) const {
    mSlots.insert(std::make_pair(++mCurrentID, slot));
    return mCurrentID;
  }

  /// Disconnects a previously connected function.
  void disconnect(int id) const {
    if (mIsIterating) {
      mSlotsToDisconnect.push_back(id);
    } else {
      mSlots.erase(id);
    }
  }

  /// Disconnects all previously connected functions.
  void disconnectAll() const {
    if (mIsIterating) {
      mDisconnectAllRequested = true;
    } else {
      mSlots.clear();
    }
  }

  /// Calls all connected functions.
  void emit(Args... p) {
    if (mIsIterating) {
      logger().warn(
          "Recursive invocation of emit! To avoid a stack overflow, the recursive invocation was "
          "skipped. Some slots might not be informed about the most recent changes.");
      return;
    }

    mIsIterating = true;

    for (auto const& it : mSlots) {
      it.second(p...);
    }

    mIsIterating = false;
    postEmitCleanUp();
  }

  /// Calls all connected functions except for one.
  void emitForAllButOne(int excludedConnectionID, Args... p) {
    if (mIsIterating) {
      logger().warn(
          "Recursive invocation of emit! To avoid a stack overflow, the recursive invocation was "
          "skipped. Some slots might not be informed about the most recent changes.");
      return;
    }

    mIsIterating = true;

    for (auto const& it : mSlots) {
      if (it.first != excludedConnectionID) {
        it.second(p...);
      }
    }

    mIsIterating = false;
    postEmitCleanUp();
  }

  /// Calls only one connected functions.
  void emitFor(int connectionID, Args... p) {
    auto const& it = mSlots.find(connectionID);
    if (it != mSlots.end()) {
      it->second(p...);
    }
  }

  /// Assignment creates new Signal.
  Signal& operator=(Signal const& other) { // NOLINT(cert-oop54-cpp)
    if (this != &other) {
      disconnectAll();
    }
    return *this;
  }

 private:
  void postEmitCleanUp() {
    if (mDisconnectAllRequested) {
      mSlots.clear();
      mDisconnectAllRequested = false;
    } else {
      for (int id : mSlotsToDisconnect) {
        mSlots.erase(id);
      }
      mSlotsToDisconnect.clear();
    }
  }

  mutable std::map<int, std::function<void(Args...)>> mSlots;
  mutable int                                         mCurrentID{0};

  mutable bool             mIsIterating            = false;
  mutable bool             mDisconnectAllRequested = false;
  mutable std::vector<int> mSlotsToDisconnect;
};

} // namespace cs::utils

#endif // CS_UTILS_SIGNAL_HPP
