////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "utils.hpp"

#include "logger.hpp"

#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <array>
#include <format>

namespace csp::wmsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string timeToString(std::string const& format, std::chrono::utc_clock::time_point time) {
  auto value = std::chrono::time_point_cast<std::chrono::milliseconds>(time);
  return std::vformat("{:" + format + "}", std::make_format_args(value));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void matchDuration(
    std::string const& input, std::regex const& re, std::chrono::utc_clock::duration& duration) {
  std::smatch match;
  std::regex_search(input, match, re);

  if (match.empty()) {
    logger().debug("Pattern does not match!");
    return;
  }

  std::vector<int> vec = {0, 0, 0, 0, 0, 0}; // years, months, days, hours, minutes, seconds

  for (size_t i = 1; i < match.size(); ++i) {
    if (match[i].matched) {
      std::string str = match[i];
      str.pop_back(); // remove last character.
      vec[i - 1] = static_cast<int>(std::stod(str));
    }
  }

  duration = std::chrono::years(vec[0]) + std::chrono::months(vec[1]) + std::chrono::days(vec[2]) +
             std::chrono::hours(vec[3]) + std::chrono::minutes(vec[4]) +
             std::chrono::seconds(vec[5]);

  if (duration.count() == 0) {
    logger().debug("Input is not valid!");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void timeDuration(std::string const& isoString, std::chrono::utc_clock::duration& duration) {
  std::regex rshort("^((?!T).)*$");

  // Check if isoString matches rshort.
  if (std::regex_match(isoString, rshort)) // no T (Time) exist
  {
    std::regex r("P([[:d:]]+Y)?([[:d:]]+M)?([[:d:]]+D)?");
    matchDuration(isoString, r, duration);
  } else {
    std::regex r("P([[:d:]]+Y)?([[:d:]]+M)?([[:d:]]+D)?T([[:d:]]+H)?([[:d:]]+M)?([[:d:]]+S|[[:d:]]+"
                 "\\.[[:d:]]+S)?");
    matchDuration(isoString, r, duration);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void convertIsoDate(std::string& date, std::chrono::utc_clock::time_point& time) {
  time = cs::utils::convert::time::toUTC(date);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void parseIsoString(std::string const& isoString, std::vector<TimeInterval>& timeIntervals) {
  std::string       timeRange;
  std::stringstream iso_stringstream(isoString);

  // Read time intervalls.
  while (std::getline(iso_stringstream, timeRange, ',')) {
    std::string       startDate;
    std::string       endDate;
    std::string       duration;
    std::stringstream timeRange_stringstream(timeRange);

    std::getline(timeRange_stringstream, startDate, '/');
    std::getline(timeRange_stringstream, endDate, '/');
    std::getline(timeRange_stringstream, duration, '/');

    TimeInterval                       tmp;
    std::chrono::utc_clock::time_point start;
    std::chrono::utc_clock::time_point end;
    convertIsoDate(startDate, start);

    // Check format of startData
    std::array<std::string, 6> formatParts = {"%Y", "-%m", "-%d", "T%H", ":%M", ":%S"};
    std::array<std::string, 6> partLengths = {"4", "2", "2", "2", "2", "2"};
    std::smatch                result;
    // Initialize result to complete startDate string
    std::regex_search(startDate, result, std::regex("^"));
    for (size_t i = 0; i < formatParts.size(); i++) {
      std::stringstream regex;
      regex << "^[\\-:.T]?[[:d:]]{" << partLengths[i] << "}";
      if (std::regex_search(
              result.suffix().first, result.suffix().second, result, std::regex(regex.str()))) {
        tmp.mFormat.append(formatParts[i]);
      } else {
        if (i == 0) {
          // No year found, using default format
          tmp.mFormat = "%Y-%m-%dT%H:%M:%SZ";
        }
        break;
      }
    }
    if (cs::utils::contains(tmp.mFormat, "T")) {
      // Time parts are present, append 'Z'
      tmp.mFormat.append("Z");
    }

    if (endDate.empty()) {
      // If there is no end date, just a single timestep.
      end = start;
    } else {
      timeDuration(duration, tmp.mSampleDuration);

      // If end date is set to current, select it according to the time format.
      if (endDate == "current") {
        auto now = std::chrono::utc_clock::now();
        if (tmp.mFormat == "%Y") {
          end = std::chrono::floor<std::chrono::years>(now);
        } else if (tmp.mFormat == "%Y-%m") {
          end = std::chrono::floor<std::chrono::months>(now);
        } else {
          end = now;
        }
      } else {
        convertIsoDate(endDate, end);
      }
    }

    tmp.mEndTime   = end;
    tmp.mStartTime = start;
    timeIntervals.push_back(tmp);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool timeInIntervals(std::chrono::utc_clock::time_point& time,
    std::vector<TimeInterval> const& timeIntervals, TimeInterval& foundInterval) {
  for (const auto& interval : timeIntervals) {
    // Check if time is within the interval
    if (time >= interval.mStartTime && time < interval.mEndTime) {
      // Calculate the number of samples since the start of the interval
      auto elapsed       = time - interval.mStartTime;
      auto samplesPassed = elapsed / interval.mSampleDuration;

      // Snap the time to the start of the current sample
      time = interval.mStartTime + samplesPassed * interval.mSampleDuration;

      // Set the found interval
      foundInterval = interval;

      return true;
    }
  }
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::optional<int>> getSizeAttribute(
    VistaXML::TiXmlElement* element, std::string const& attributeName) {
  std::optional<int> value = getAttribute<int>(element, attributeName);
  if (value.has_value()) {
    std::optional<int> inner;
    if (value.value() != 0) {
      inner = value.value();
    }
    return inner;
  }
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
std::optional<std::string> getAttribute<std::string>(
    VistaXML::TiXmlElement* element, std::string const& attributeName) {
  const std::string* result = element->Attribute(attributeName);
  if (result != nullptr) {
    return *result;
  }
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
std::optional<bool> getAttribute<bool>(
    VistaXML::TiXmlElement* element, std::string const& attributeName) {
  std::optional<int> value = getAttribute<int>(element, attributeName);
  if (value.has_value()) {
    return value.value() == 1;
  }
  std::optional<std::string> valueStr = getAttribute<std::string>(element, attributeName);
  if (valueStr.has_value()) {
    if (valueStr.value() == "1" || valueStr.value() == "true") {
      return true;
    }
    if (valueStr.value() == "0" || valueStr.value() == "false") {
      return false;
    }
  }
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace utils

} // namespace csp::wmsoverlays
