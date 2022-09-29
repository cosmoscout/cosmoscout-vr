////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_OBSERVABLE_MAP_HPP
#define CS_UTILS_OBSERVABLE_MAP_HPP

#include "Signal.hpp"

#include <unordered_map>

namespace cs::utils {

/// This is a wrapper around a std::unordered_map which adds two Signals: onAdd and onRemove. It
/// only exposes const access to the contained objects, so they are basically immutable. If a
/// property of an item needs to changed, the respective item needs to be removed and re-added.
template <typename K, typename V>
class ObservableMap {
 public:
  /// This signal will be emitted right after an item was added to the map.
  Signal<K, V> const& onAdd() const {
    return mOnAdd;
  }

  /// This signal will be emitted right after an item was removed from the map.
  Signal<K, V> const& onRemove() const {
    return mOnRemove;
  };

  /// Inserts a new element into the container if there is no element with the key in the container.
  void insert(K key, V value) {
    auto res = mMap.emplace(std::move(key), std::move(value));

    if (res.second) {
      mOnAdd.emit(res.first->first, res.first->second);
    }
  }

  /// Finds an element with key equivalent to key.
  typename std::unordered_map<K, V>::const_iterator find(K const& key) const {
    return mMap.find(key);
  }

  /// Returns a reference to the mapped value of the element with key equivalent to key. If no such
  /// element exists, an exception of type std::out_of_range is thrown.
  V const& at(K const& key) const {
    return mMap.at(key);
  }

  /// Removes the specified element from the container.
  void erase(K const& key) {
    auto item = mMap.extract(key);

    if (item) {
      mOnRemove.emit(item.key(), item.value());
    }
  }

  /// Erases all elements from the container. The onRemove signal will be emitted once for each
  /// element.
  void clear() {
    for (auto const& [key, value] : mMap) {
      mOnRemove.emit(key, value);
    }
    mMap.clear();
  }

  /// Checks if the container has no elements.
  bool empty() const {
    return mMap.empty();
  }

  /// Returns the number of elements in the container.
  typename std::unordered_map<K, V>::size_type size() const {
    return mMap.size();
  }

  /// Iterator API.
  typename std::unordered_map<K, V>::const_iterator begin() const {
    return mMap.begin();
  }

  typename std::unordered_map<K, V>::const_iterator end() const {
    return mMap.end();
  }

 private:
  Signal<K, V>             mOnAdd;
  Signal<K, V>             mOnRemove;
  std::unordered_map<K, V> mMap;
};

} // namespace cs::utils

#endif // CS_UTILS_OBSERVABLE_MAP_HPP
