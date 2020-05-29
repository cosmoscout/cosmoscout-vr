////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WMS_UTILS_HPP
#define CSP_WMS_UTILS_HPP

#include "../../../src/cs-utils/convert.hpp"

#include <optional>
#include <regex>

namespace csp::simplewmsbodies {

/// Struct of timeintervals of the data set.
struct TimeInterval {
  boost::posix_time::ptime mStartTime;        ///< The beginning of the interval.
  boost::posix_time::ptime mEndTime;          ///< The end of the interval.
  std::string              mFormat;           ///< The string format of time values.
  int                      mIntervalDuration; ///< The duration of the interval in seconds.
};

namespace utils {

/// Create formatted date, time string from time value.
std::string timeToString(std::string const& format, boost::posix_time::ptime time);

/// Match years, months, days, etc. in regex input string and calculate duration.
void matchDuration(std::string const& input, std::regex const& re, int& duration);

/// Determine time format and interval duration from string regex.
void timeDuration(std::string const& isoString, int& duration, std::string& format);

/// Convert date from string to time.
void convertIsoDate(std::string& date, boost::posix_time::ptime& time);

/// Parse time intervals from string.
void parseIsoString(std::string const& isoString, std::vector<TimeInterval>& timeIntervals);

/// Check whether the given time is inside one of the time intervals.
bool timeInIntervals(boost::posix_time::ptime time, std::vector<TimeInterval>& timeIntervals,
    boost::posix_time::time_duration& timeSinceStart, int& intervalDuration, std::string& format);

} // namespace utils

} // namespace csp::simplewmsbodies

#endif // CSP_WMS_UTILS_HPP