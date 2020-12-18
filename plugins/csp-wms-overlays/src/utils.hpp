////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WMS_UTILS_HPP
#define CSP_WMS_UTILS_HPP

#include "../../../src/cs-utils/convert.hpp"

#include <VistaTools/tinyXML/tinyxml.h>

#include <optional>
#include <regex>

namespace csp::wmsoverlays {

/// Struct for the duration of the WMS time step.
///	Ideally only one of the members should be non-zero.
struct Duration {
  int                              mYears        = 0;
  int                              mMonths       = 0;
  boost::posix_time::time_duration mTimeDuration = boost::posix_time::seconds(0);

  bool isDuration() const;
};

/// Struct of timeintervals of the data set.
struct TimeInterval {
  boost::posix_time::ptime mStartTime;      ///< The beginning of the interval.
  boost::posix_time::ptime mEndTime;        ///< The end of the interval.
  std::string              mFormat;         ///< The string format of time values.
  Duration                 mSampleDuration; ///< The duration of one sample in WMS interval.
};

namespace utils {

/// Create formatted date, time string from time value.
std::string timeToString(std::string const& format, boost::posix_time::ptime time);

/// Match years, months, days, etc. in regex input string and calculate duration.
void matchDuration(std::string const& input, std::regex const& re, Duration& duration);

/// Determine time format and interval duration from string regex.
void timeDuration(std::string const& isoString, Duration& duration, std::string& format);

/// Convert date from string to time.
void convertIsoDate(std::string& date, boost::posix_time::ptime& time);

/// Parse time intervals from string.
void parseIsoString(std::string const& isoString, std::vector<TimeInterval>& timeIntervals);

/// Check whether the given time is inside one of the time intervals.
/// Then calculate the start time of the current sample if it is in the interval.
bool timeInIntervals(boost::posix_time::ptime& time, std::vector<TimeInterval>& timeIntervals,
    Duration& sampleDuration, std::string& format);

/// Adds the interval duration to the given time.
/// The duration can be either in years, months or in time_duration.
/// Adds the interval multiple times, if it is specified (for e.g. for pre-fetch).
boost::posix_time::ptime addDurationToTime(
    boost::posix_time::ptime time, Duration duration, int multiplier = 1);

/// Tries to get the text of a XML element.
/// Starts at baseElement and then descends into the children given as childPath.
/// The return value is empty if the requested element is not present.
std::optional<std::string> getElementText(
    VistaXML::TiXmlElement* baseElement, std::vector<std::string> childPath);

/// Tries to get the value of a boolean attribute on the given element.
/// If the attribute contains an integer the values will be mapped to booleans as follows:
/// 0->false, 1->true
/// The return value is empty if the requested element is not present.
std::optional<bool> getBoolAttribute(VistaXML::TiXmlElement* element, std::string attributeName);

/// Tries to get the value of an integer attribute representing a size.
/// The returned value (inner optional) is empty if the attribute specifies an unlimited size.
/// The return value (outer optional) is empty if the requested attribute is not present.
std::optional<std::optional<int>> getSizeAttribute(
    VistaXML::TiXmlElement* element, std::string attributeName);

/// Gets the value of the given attribute on the given element.
/// The return value is empty if the requested element is not present.
template <typename T>
std::optional<T> getAttribute(VistaXML::TiXmlElement* element, std::string attributeName) {
  T   value;
  int result = element->QueryValueAttribute<T>(attributeName, &value);
  if (result == VistaXML::TIXML_SUCCESS) {
    return value;
  }
  return {};
}

/// Sets the given var to the value of the optional, if it is present.
template <typename T>
void setOrKeep(T& var, std::optional<T> optional) {
  var = optional.value_or(var);
}

/// Sets the given var to the value of the optional, if it is present.
template <typename T>
void setOrKeep(std::optional<T>& var, std::optional<T> optional) {
  if (optional.has_value()) {
    var = optional.value();
  }
}

std::optional<double> optstod(std::optional<std::string> string);

} // namespace utils

} // namespace csp::wmsoverlays

#endif // CSP_WMS_UTILS_HPP