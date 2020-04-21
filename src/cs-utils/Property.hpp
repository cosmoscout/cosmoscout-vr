////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_PROPERTY_HPP
#define CS_UTILS_PROPERTY_HPP

#include "Signal.hpp"

#include <iostream>

namespace cs::utils {

/// A Property encapsulates a value and may inform you on any changes applied to this value.
/// All functions given to connect() will be called when the internal value is about to be changed.
/// The new value is passed as parameter, to access the old value you can use the get() method, as
/// the internal value will be overwritten after the signal emission. Const-correctness in this
/// class is relative to the contained value: All methods which are marked const are guaranteed not
/// to change the internal value.
template <typename T>
class Property {

 public:
  using value_type = T;

  /// Properties for built-in types are automatically initialized to 0.
  Property() = default;

  Property(T val) // NOLINT(hicpp-explicit-conversions)
      : mValue(std::move(val)) {
  }

  Property(Property<T> const& other)
      : mValue(other.mValue) {
  }

  Property(Property<T>&& other) noexcept
      : mOnChange(std::move(other.mOnChange))
      , mConnection(other.mConnection)
      , mConnectionID(other.mConnectionID)
      , mValue(other.mValue) {
  }

  Property& operator=(Property<T>&& other) noexcept {
    if (this != &other) {
      mOnChange     = std::move(other.mOnChange);
      mConnection   = other.mConnection;
      mConnectionID = other.mConnectionID;
      mValue        = other.mValue;
    }

    return *this;
  }

  virtual ~Property() {
    if (mConnection) {
      mConnection->disconnect(mConnectionID);
    }
  };

  /// The given function is called when the internal value is about to be changed. The new value
  /// is passed as parameter, to access the old value you can use the get() method, as the internal
  /// value will be overwritten after the signal emission.
  virtual int connect(std::function<void(T)> const& slot) const {
    return mOnChange.connect(slot);
  }

  /// Same as above, but in addition, the given function is immediately called once.
  virtual int connectAndTouch(std::function<void(T)> const& slot) const {
    int connection = mOnChange.connect(slot);
    touch(connection);
    return connection;
  }

  /// Connects two Properties to each other. If the source's value is changed, this' value will be
  /// changed as well.
  virtual void connectFrom(Property<T> const& source) {
    disconnect();
    mConnection   = &source;
    mConnectionID = source.connect([this](T const& value) {
      set(value);
      return true;
    });
    set(source.get());
  }

  /// If this Property is connected from another property, it will be disconnected.
  virtual void disconnect() const {
    if (mConnection) {
      mConnection->disconnect(mConnectionID);
      mConnectionID = -1;
      mConnection   = nullptr;
    }
  }

  /// Disconnects a previously connected function.
  void disconnect(int id) const {
    mOnChange.disconnect(id);
  }

  /// If there are any Properties connected to this Property, they won't be notified of any further
  /// changes.
  virtual void disconnectAll() const {
    mOnChange.disconnectAll();
  }

  /// Sets the Property to a new value. onChange() will be emitted.
  virtual void set(T const& value) {
    if (value != mValue) {
      mOnChange.emit(value);
      mValue = value;
    }
  }

  /// Sets the Property to a new value. onChange() will be emitted for all but one connections.
  virtual void setWithEmitForAllButOne(T const& value, int excludeConnection) {
    if (value != mValue) {
      mOnChange.emitForAllButOne(excludeConnection, value);
      mValue = value;
    }
  }

  /// Sets the Property to a new value. onChange() will not be emitted.
  void setWithNoEmit(T const& value) {
    mValue = value;
  }

  /// Emits onChange() even if the value did not change.
  void touch() const {
    mOnChange.emit(mValue);
  }

  /// Emits onChange() even if the value did not change but only for one connection.
  void touch(int connection) const {
    mOnChange.emitFor(connection, mValue);
  }

  /// Returns the value.
  virtual T const& get() const {
    return mValue;
  }

  /// Assigns the value of another Property.
  Property<T>& operator=(Property<T> const& rhs) {
    set(rhs.get());
    return *this;
  }

  /// Assigns a new value to this Property.
  virtual Property<T>& operator=(T const& rhs) {
    set(rhs);
    return *this;
  }

  /// Compares the values of two Properties.
  bool operator==(Property<T> const& rhs) const {
    return get() == rhs.get();
  }
  bool operator!=(Property<T> const& rhs) const {
    return get() != rhs.get();
  }

  /// Compares the values of the Property to another value.
  bool operator==(T const& rhs) const {
    return get() == rhs;
  }
  bool operator!=(T const& rhs) const {
    return get() != rhs;
  }

  /// Returns the value of this Property.
  T const& operator()() const {
    return get();
  }

 protected:
  mutable Signal<T> mOnChange;

  mutable Property<T> const* mConnection{nullptr};
  mutable int                mConnectionID{-1};
  T                          mValue{}; // Default initialize (primitives => 0 | false).
};

/// Stream operators.
template <typename T>
std::ostream& operator<<(std::ostream& out_stream, Property<T> const& val) {
  out_stream << val.get();
  return out_stream;
}

template <typename T>
std::istream& operator>>(std::istream& in_stream, Property<T>& val) {
  T tmp;
  in_stream >> tmp;
  val.set(tmp);
  return in_stream;
}

} // namespace cs::utils

#endif // CS_UTILS_PROPERTY_HPP
