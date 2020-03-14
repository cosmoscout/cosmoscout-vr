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

namespace cs::utils {

/// A signal object may call multiple slots with the same signature. You can connect functions to
/// the signal which will be called when the emit() method on the signal object is invoked. Any
/// argument passed to emit() will be passed to the given functions.
template <typename... Args>
class Signal {

 public:
  Signal()
      : mCurrentID(0) {
  }

  /// Copy creates new signal.
  Signal(Signal const& other)
      : mCurrentID(0) {
  }

  Signal(Signal&& other)
      : mSlots(std::move(other.mSlots))
      , mCurrentID(other.mCurrentID) {
  }

  /// Connects a member function to this Signal.
  template <typename T>
  int connectMember(T* inst, void (T::*func)(Args...)) {
    return connect([=](Args... args) { (inst->*func)(args...); });
  }

  /// Connects a const member function to this Signal.
  template <typename T>
  int connectMember(T* inst, void (T::*func)(Args...) const) {
    return connect([=](Args... args) { (inst->*func)(args...); });
  }

  /// Connects a std::function to the signal. The returned value can be used to disconnect the
  /// function again.
  int connect(std::function<void(Args...)> const& slot) const {
    mSlots.insert(std::make_pair(++mCurrentID, slot));
    return mCurrentID;
  }

  /// Disconnects a previously connected function.
  void disconnect(int id) const {
    mSlots.erase(id);
  }

  /// Disconnects all previously connected functions.
  void disconnectAll() const {
    mSlots.clear();
  }

  /// Calls all connected functions.
  void emit(Args... p) {
    for (auto it : mSlots) {
      it.second(p...);
    }
  }

  /// Calls all connected functions except for one.
  void emitForAllButOne(int excludedConnectionID, Args... p) {
    for (auto it : mSlots) {
      if (it.first != excludedConnectionID) {
        it.second(p...);
      }
    }
  }

  /// Calls only one connected functions.
  void emitFor(int connectionID, Args... p) {
    auto const& it = mSlots.find(connectionID);
    if (it != mSlots.end()) {
      it.second(p...);
    }
  }

  /// Assignment creates new Signal.
  Signal& operator=(Signal const& other) {
    disconnectAll();

    return *this;
  }

 private:
  mutable std::map<int, std::function<void(Args...)>> mSlots;
  mutable int                                         mCurrentID;
};

} // namespace cs::utils

#endif // CS_UTILS_SIGNAL_HPP
