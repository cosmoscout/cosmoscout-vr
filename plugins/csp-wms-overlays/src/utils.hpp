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

} // namespace utils

} // namespace csp::wmsoverlays

#endif // CSP_WMS_UTILS_HPP