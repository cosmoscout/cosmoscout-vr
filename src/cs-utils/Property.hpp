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
/// All functions given to connect() will be called when the internal value is about to be changed. The new value
  /// is passed as parameter, to access the old value you can use the get() method, as the internal
  /// value will be overwritten after the signal emission.
  /// Const-correctness in this class is relative to the contained value: All methods which are marked const are guaranteed not to change the internal value.
template <typename T>
class Property {

 public:
  typedef T value_type;

  /// Properties for built-in types are automatically initialized to 0.
  Property()
      : mConnection(nullptr)
      , mConnectionID(-1) {
  }

  Property(T const& val)
      : mValue(val)
      , mConnection(nullptr)
      , mConnectionID(-1) {
  }

  Property(T&& val)
      : mValue(std::move(val))
      , mConnection(nullptr)
      , mConnectionID(-1) {
  }

  Property(Property<T> const& other)
      : mValue(other.mValue)
      , mConnection(nullptr)
      , mConnectionID(-1) {
  }

  Property(Property<T>&& other)
      : mOnChange(std::move(other.mOnChange))
      , mConnection(other.mConnection)
      , mConnectionID(other.mConnectionID)
      , mValue(other.mValue) {
  }

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
  virtual void setWithEmitForAllButOne(T const& value, int excludeConnection = -1) {
    if (value != mValue) {
      mOnChange.emitForAllButOne(excludeConnection, mValue);
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
  virtual Property<T>& operator=(Property<T> const& rhs) {
    set(rhs.mValue);
    return *this;
  }

  /// Assigns a new value to this Property.
  virtual Property<T>& operator=(T const& rhs) {
    set(rhs);
    return *this;
  }

  /// Compares the values of two Properties.
  bool operator==(Property<T> const& rhs) const {
    return Property<T>::get() == rhs.get();
  }
  bool operator!=(Property<T> const& rhs) const {
    return Property<T>::get() != rhs.get();
  }

  /// Compares the values of the Property to another value.
  bool operator==(T const& rhs) const {
    return Property<T>::get() == rhs;
  }
  bool operator!=(T const& rhs) const {
    return Property<T>::get() != rhs;
  }

  /// Returns the value of this Property.
  T const& operator()() const {
    return Property<T>::get();
  }

 private:
  mutable Signal<T> mOnChange;

  mutable Property<T> const* mConnection;
  mutable int                mConnectionID;
  T                          mValue;
};

/// Specialization for built-in default constructors.
template <>
inline Property<double>::Property()
    : mConnection(nullptr)
    , mConnectionID(-1)
    , mValue(0.0) {
}

template <>
inline Property<float>::Property()
    : mConnection(nullptr)
    , mConnectionID(-1)
    , mValue(0.f) {
}

template <>
inline Property<short>::Property()
    : mConnection(nullptr)
    , mConnectionID(-1)
    , mValue(0) {
}

template <>
inline Property<int>::Property()
    : mConnection(nullptr)
    , mConnectionID(-1)
    , mValue(0) {
}

template <>
inline Property<char>::Property()
    : mConnection(nullptr)
    , mConnectionID(-1)
    , mValue(0) {
}

template <>
inline Property<unsigned>::Property()
    : mConnection(nullptr)
    , mConnectionID(-1)
    , mValue(0) {
}

template <>
inline Property<bool>::Property()
    : mConnection(nullptr)
    , mConnectionID(-1)
    , mValue(false) {
}

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
