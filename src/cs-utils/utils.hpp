////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_UTILS_HPP
#define CS_UTILS_UTILS_HPP

#include "cs_utils_export.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

/// Utility functions for all sorts of stuff.
namespace cs::utils {

/// Defines the order in which objects will be rendered.
enum class CS_UTILS_EXPORT DrawOrder : int {
  eClearHDRBuffer   = 100,
  ePlanets          = 200,
  eOpaqueItems      = 300,
  eStars            = 400,
  eAtmospheres      = 500,
  eToneMapping      = 600,
  eTransparentItems = 700,
  eRay              = 800,
  eGui              = 900
};
template <typename T>
bool contains(T const& container, typename T::value_type const& item) {
  return std::find(std::begin(container), std::end(container), item) != std::end(container);
}

template <typename K, typename V>
bool contains(std::map<K, V> const& map, K const& key) {
  return map.find(key) != map.end();
}

template <typename K, typename V>
bool contains(std::unordered_map<K, V> const& map, K const& key) {
  return map.find(key) != map.end();
}

/// Replaces all occurrences of the sPlaceHolder parameter in sInput by sNewString.
CS_UTILS_EXPORT void replaceString(
    std::string& sInput, std::string const& sPlaceHolder, std::string const& sNewString);

template <typename T>
std::string toString(T const& v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

template <>
CS_UTILS_EXPORT std::string toString(float const& v);

template <>
CS_UTILS_EXPORT std::string toString(double const& v);

template <>
CS_UTILS_EXPORT std::string toString(std::string const& v);

CS_UTILS_EXPORT std::string toString(char const* v);

template <typename T>
T fromString(std::string const& v) {
  std::istringstream iss(v);
  T                  r;
  iss >> r;
  return r;
}

/// Splits the given string into chunks separated by the delim character.
CS_UTILS_EXPORT std::vector<std::string> splitString(std::string const& s, char delim);

/// A template to cast an enum class to its underlying type.
template <typename T>
constexpr typename std::underlying_type<T>::type enumCast(T val) {
  return static_cast<typename std::underlying_type<T>::type>(val);
}

/// Well, does what is says.
float CS_UTILS_EXPORT getCurrentFarClipDistance();

} // namespace cs::utils

#endif // CS_UTILS_UTILS_HPP
