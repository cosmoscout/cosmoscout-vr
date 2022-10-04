////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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

/// These macros can be used to selectively disable specific gcc / clang or msvc warnings.
#if defined(__clang__) || defined(__GNUC__)
#define CS_DO_PRAGMA(X) _Pragma(#X)
#define CS_WARNINGS_PUSH _Pragma("GCC diagnostic push")
#define CS_WARNINGS_POP _Pragma("GCC diagnostic pop")
#define CS_DISABLE_GCC_WARNING(warningName) CS_DO_PRAGMA(GCC diagnostic ignored warningName)
#define CS_DISABLE_MSVC_WARNING(warningNumber)
#elif defined(_MSC_VER)
#define CS_WARNINGS_PUSH __pragma(warning(push))
#define CS_WARNINGS_POP __pragma(warning(pop))
#define CS_DISABLE_MSVC_WARNING(warningNumber) __pragma(warning(disable : warningNumber))
#define CS_DISABLE_GCC_WARNING(warningName)
#endif

/// Utility functions for all sorts of stuff.
namespace cs::utils {

/// Defines the order in which objects will be rendered.
enum class CS_UTILS_EXPORT DrawOrder : int {
  eSetupOpenGL      = 0,
  eClearHDRBuffer   = 100,
  eStars            = 200,
  ePlanets          = 300,
  eOpaqueItems      = 400,
  eAtmospheres      = 500,
  eToneMapping      = 600,
  eOpaqueNonHDR     = 650,
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

/// Returns true if the given string contains at least once the given value.
CS_UTILS_EXPORT bool contains(std::string const& string, std::string const& value);

/// Returns true if the start of the given string fully contains the given prefix.
CS_UTILS_EXPORT bool startsWith(std::string const& string, std::string const& prefix);

/// Returns true if the end of the given string fully contains the given postfix.
CS_UTILS_EXPORT bool endsWith(std::string const& string, std::string const& postfix);

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

/// Executes a system command and returns the output.
std::string exec(std::string const& cmd);

/// Can be used to check the operating system at compile time.
enum class OS { eLinux, eWindows };

#ifdef __linux__
constexpr OS HostOS = OS::eLinux;
#elif _WIN32
constexpr OS HostOS = OS::eWindows;
#endif

} // namespace cs::utils

#endif // CS_UTILS_UTILS_HPP
