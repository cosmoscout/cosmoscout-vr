////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_DEFAULT_PROPERTY_HPP
#define CS_UTILS_DEFAULT_PROPERTY_HPP

#include "Property.hpp"

#include <iostream>

namespace cs::utils {

/// A Property encapsulates a value and may inform you on any changes applied to this value. This
/// DefaultProperty stores an additional constant default value. There are methods to check whether
/// the DefaultValue is in the default state and it can be resetted to the default state.
template <typename T>
class DefaultProperty : public Property<T> {
 public:
  /// There is no default constructor, as a default value has to be given at construction time.
  explicit DefaultProperty(T const& val)
      : Property<T>(val)
      , mDefaultValue(val) {
  }

  explicit DefaultProperty(T&& val)
      : Property<T>(val)
      , mDefaultValue(std::move(val)) {
  }

  DefaultProperty(DefaultProperty<T> const& other)
      : Property<T>(other.mValue)
      , mDefaultValue(other.mDefaultValue) {
  }

  DefaultProperty(DefaultProperty<T>&& other) noexcept
      : Property<T>(std::move(other.mValue))
      , mDefaultValue(std::move(other.mDefaultValue)) {
  }

  ~DefaultProperty() override = default;

  /// Returns true if the current value of the Property equals to the default state.
  bool isDefault() const {
    return Property<T>::get() == mDefaultValue;
  }

  /// Resets the Property to its default state.
  void reset() {
    Property<T>::set(mDefaultValue);
  }

  /// Assigns the value of another DefaultProperty. The internal default value is not changed.
  DefaultProperty<T>& operator=(DefaultProperty<T> const& rhs) {
    Property<T>::set(rhs.get());
    return *this;
  }

  /// Assigns the value of another DefaultProperty. The internal default value is not changed.
  DefaultProperty<T>& operator=(DefaultProperty<T>&& rhs) noexcept {
    Property<T>::set(rhs.get());
    return *this;
  }

  /// Assigns a new value to this DefaultProperty.
  DefaultProperty<T>& operator=(T const& rhs) override {
    Property<T>::set(rhs);
    return *this;
  }

  /// Compares the values of two Properties.
  bool operator==(DefaultProperty<T> const& rhs) const {
    return Property<T>::get() == rhs.get() && mDefaultValue == rhs.mDefaultValue;
  }

  bool operator!=(DefaultProperty<T> const& rhs) const {
    return Property<T>::get() != rhs.get() || mDefaultValue != rhs.mDefaultValue;
  }

 private:
  const T mDefaultValue;
};

} // namespace cs::utils

#endif // CS_UTILS_DEFAULT_PROPERTY_HPP
