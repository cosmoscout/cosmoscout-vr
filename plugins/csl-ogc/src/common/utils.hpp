////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_UTILS_HPP
#define CSL_OGC_UTILS_HPP

#include "csl_ogc_export.hpp"

#include "../../../src/cs-utils/convert.hpp"

#include <VistaTools/tinyXML/tinyxml.h>

#include <nlohmann/json.hpp>

#include <optional>
#include <regex>

namespace csl::ogc {

/// Struct for storing a geographical bounding box for a map.
/// Coordinates should be given in degrees.
struct CSL_OGC_EXPORT Bounds {
  double mMinLon{-180.};
  double mMaxLon{180.};
  double mMinLat{-90.};
  double mMaxLat{90.};

  Bounds() = default;

  Bounds(double minLon, double maxLon, double minLat, double maxLat)
      : mMinLon(minLon)
      , mMaxLon(maxLon)
      , mMinLat(minLat)
      , mMaxLat(maxLat) {
  }

  inline bool operator!=(const Bounds& rhs) const {
    return mMinLon != rhs.mMinLon || mMaxLon != rhs.mMaxLon || mMinLat != rhs.mMinLat ||
           mMaxLat != rhs.mMaxLat;
  }

  inline bool operator==(const Bounds& rhs) const {
    return mMinLon == rhs.mMinLon && mMaxLon == rhs.mMaxLon && mMinLat == rhs.mMinLat &&
           mMaxLat == rhs.mMaxLat;
  }
};

void CSL_OGC_EXPORT from_json(nlohmann::json const& j, Bounds& o);
void CSL_OGC_EXPORT to_json(nlohmann::json& j, Bounds const& o);

/// Struct for the duration of the WMS time step.
/// Ideally only one of the members should be non-zero.
struct CSL_OGC_EXPORT Duration {
  int                              mYears        = 0;
  int                              mMonths       = 0;
  boost::posix_time::time_duration mTimeDuration = boost::posix_time::seconds(0);

  /// Checks whether the object represents a non-zero duration.
  bool isDuration() const;

  inline bool operator==(const Duration& rhs) const {
    return mYears == rhs.mYears && mMonths == rhs.mMonths && mTimeDuration == rhs.mTimeDuration;
  }
};

/// Struct of timeintervals of the data set.
struct CSL_OGC_EXPORT TimeInterval {
  boost::posix_time::ptime mStartTime;      ///< The beginning of the interval.
  boost::posix_time::ptime mEndTime;        ///< The end of the interval.
  std::string              mFormat;         ///< The string format of time values.
  Duration                 mSampleDuration; ///< The duration of one sample in WMS interval.

  inline bool operator==(const TimeInterval& rhs) const {
    return mStartTime == rhs.mStartTime && mEndTime == rhs.mEndTime && mFormat == rhs.mFormat &&
           mSampleDuration == rhs.mSampleDuration;
  }
  inline bool operator!=(const TimeInterval& rhs) const {
    return !(*this == rhs);
  }
};

/// This namespace contains some utility functions for:
/// A) Handling the format used by WMS for describing temporal data
///     The valid times for a given layer are generally specified as a comma-seperated list of
///     either single points in time or of temporal ranges expressed using the syntax
///     'start/end/period', where start and end are points in time and period is given using the
///     duration format specified by ISO 8601.
///     Points in time are specified using the ISO 8601:2000 "extended" format. Because
///     least-significant digits may be omitted for data of low temporal precision, the functions in
///     cs::utils::convert::time can not be used here. Some servers only accept times of the same
///     precision as the one given in the layer capabilities, so the precision is saved as a format
///     string in TimeInterval objects.
/// B) More convenient parsing of XML documents
///     Adds some function templates for getting generic types from XML elements. Uses
///     std::optionals as return values to signify whether a value was found or not.
namespace utils {

/// Create formatted date, time string from time value.
std::string CSL_OGC_EXPORT timeToString(std::string const& format, boost::posix_time::ptime time);

/// Match years, months, days, etc. in regex input string and calculate duration.
void CSL_OGC_EXPORT matchDuration(
    std::string const& input, std::regex const& re, Duration& duration);

/// Determine interval duration from string regex.
void CSL_OGC_EXPORT timeDuration(std::string const& isoString, Duration& duration);

/// Convert date from string to time.
void CSL_OGC_EXPORT convertIsoDate(std::string& date, boost::posix_time::ptime& time);

/// Parse time intervals from string.
void CSL_OGC_EXPORT parseIsoString(
    std::string const& isoString, std::vector<TimeInterval>& timeIntervals);

/// Check whether the given time is inside one of the time intervals.
/// Then calculate the start time of the current sample if it is in the interval.
/// If the time is in an interval, the 'foundInterval' parameter will be set to it.
bool CSL_OGC_EXPORT timeInIntervals(boost::posix_time::ptime& time,
    std::vector<TimeInterval> const& timeIntervals, TimeInterval& foundInterval);

/// Adds the interval duration to the given time.
/// The duration can be either in years, months or in time_duration.
/// Adds the interval multiple times, if it is specified (e.g. for pre-fetch).
boost::posix_time::ptime CSL_OGC_EXPORT addDurationToTime(
    boost::posix_time::ptime time, Duration const& duration, int multiplier = 1);

/// Tries to get the value contained in a XML element.
/// Starts at baseElement and then descends into the children given as childPath.
/// The return value is empty if the requested element is not present.
template <typename T>
std::optional<T> getElementValue(
    VistaXML::TiXmlElement* baseElement, std::vector<std::string> const& childPath = {}) {
  VistaXML::TiXmlHandle elementHandle(baseElement);
  for (std::string const& child : childPath) {
    elementHandle = elementHandle.FirstChildElement(child);
  }
  VistaXML::TiXmlElement* element = elementHandle.ToElement();
  if (element != nullptr && element->FirstChild() != nullptr) {
    if constexpr (std::is_same_v<T, std::string>) {
      return element->FirstChild()->ValueStr();
    } else {
      std::stringstream text;
      text << element->FirstChild()->ValueStr();
      T value;
      if (text >> value) {
        return value;
      }
    }
  }
  return {};
}

/// Tries to get the value of an integer attribute representing a size.
/// The returned value (inner optional) is empty if the attribute specifies an unlimited size.
/// The return value (outer optional) is empty if the requested attribute is not present.
std::optional<std::optional<int>> CSL_OGC_EXPORT getSizeAttribute(
    VistaXML::TiXmlElement* element, std::string const& attributeName);

/// Gets the value of the given attribute on the given element.
/// The return value is empty if the requested element is not present.
template <typename T>
std::optional<T> getAttribute(
    VistaXML::TiXmlElement const* element, std::string const& attributeName) {
  T   value;
  int result = element->QueryValueAttribute<T>(attributeName, &value);
  if (result == VistaXML::TIXML_SUCCESS) {
    return value;
  }
  return {};
}

/// QueryValueAttribute does not work for strings containing spaces, so Attribute is used instead.
template <>
std::optional<std::string> CSL_OGC_EXPORT getAttribute<std::string>(
    VistaXML::TiXmlElement const* element, std::string const& attributeName);

/// Booleans may be given as either integers (0->false, 1->true) or strings.
template <>
std::optional<bool> CSL_OGC_EXPORT getAttribute<bool>(
    VistaXML::TiXmlElement const* element, std::string const& attributeName);

/// Sets the given var to the value of the optional, if it is present.
template <typename T>
void setOrKeep(T& var, std::optional<T> const& optional) {
  var = optional.value_or(var);
}

/// Sets the given var to the value of the optional, if it is present.
template <typename T>
void setOrKeep(std::optional<T>& var, std::optional<T> const& optional) {
  if (optional.has_value()) {
    var = optional.value();
  }
}

/// Splits a string at each delimiter position
template <typename Out>
void split(const std::string& s, char delim, Out result) {
  std::istringstream iss(s);
  std::string        item;
  while (std::getline(iss, item, delim)) {
    *result++ = item;
  }
}

std::vector<std::string> split(const std::string& s, char delim);

} // namespace utils

} // namespace csl::ogc

#endif // CSL_OGC_UTILS_HPP
