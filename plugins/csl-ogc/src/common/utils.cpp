////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "utils.hpp"

#include "../logger.hpp"

#include "../../../src/cs-utils/utils.hpp"

#include "../../../../src/cs-core/Settings.hpp"

namespace csl::ogc {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Bounds2D& o) {
  if (j.is_array()) {
    std::array<double, 4> bounds{};
    j.get_to(bounds);
    o.mMinLon = bounds[0];
    o.mMaxLon = bounds[1];
    o.mMinLat = bounds[2];
    o.mMaxLat = bounds[3];
  } else {
    cs::core::Settings::deserialize(j, "minLon", o.mMinLon);
    cs::core::Settings::deserialize(j, "maxLon", o.mMaxLon);
    cs::core::Settings::deserialize(j, "minLat", o.mMinLat);
    cs::core::Settings::deserialize(j, "maxLat", o.mMaxLat);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void to_json(nlohmann::json& j, const Bounds2D& o) {
  cs::core::Settings::serialize(j, "minLon", o.mMinLon);
  cs::core::Settings::serialize(j, "maxLon", o.mMaxLon);
  cs::core::Settings::serialize(j, "minLat", o.mMinLat);
  cs::core::Settings::serialize(j, "maxLat", o.mMaxLat);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Bounds3D& o) {
  if (j.is_array()) {
    std::array<double, 6> bounds{};
    j.get_to(bounds);
    o.mMinLon    = bounds[0];
    o.mMaxLon    = bounds[1];
    o.mMinLat    = bounds[2];
    o.mMaxLat    = bounds[3];
    o.mMinHeight = bounds[4];
    o.mMaxHeight = bounds[5];
  } else {
    cs::core::Settings::deserialize(j, "minLon", o.mMinLon);
    cs::core::Settings::deserialize(j, "maxLon", o.mMaxLon);
    cs::core::Settings::deserialize(j, "minLat", o.mMinLat);
    cs::core::Settings::deserialize(j, "maxLat", o.mMaxLat);
    cs::core::Settings::deserialize(j, "minHeight", o.mMinHeight);
    cs::core::Settings::deserialize(j, "maxHeight", o.mMaxHeight);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void to_json(nlohmann::json& j, const Bounds3D& o) {
  cs::core::Settings::serialize(j, "minLon", o.mMinLon);
  cs::core::Settings::serialize(j, "maxLon", o.mMaxLon);
  cs::core::Settings::serialize(j, "minLat", o.mMinLat);
  cs::core::Settings::serialize(j, "maxLat", o.mMaxLat);
  cs::core::Settings::serialize(j, "minHeight", o.mMinHeight);
  cs::core::Settings::serialize(j, "maxHeight", o.mMaxHeight);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Duration::isDuration() const {
  return mYears + mMonths + mTimeDuration.total_seconds() != 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string timeToString(std::string const& format, boost::posix_time::ptime time) {
  std::stringstream sstr;
  // Delete is called by the std::locale constructor.
  auto* facet = new boost::posix_time::time_facet();
  facet->format(format.c_str());
  sstr.imbue(std::locale(std::locale::classic(), facet));
  sstr << time;

  return sstr.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void matchDuration(std::string const& input, std::regex const& re, Duration& duration) {
  std::smatch match;
  std::regex_search(input, match, re);

  if (match.empty()) {
    spdlog::debug("Pattern does not match!");
    return;
  }

  std::vector vec = {0, 0, 0, 0, 0, 0}; // years, months, days, hours, minutes, seconds

  for (size_t i = 1; i < match.size(); ++i) {
    if (match[i].matched) {
      std::string str = match[i];
      str.pop_back(); // remove last character.
      vec[i - 1] = static_cast<int>(std::stod(str));
    }
  }

  duration.mYears        = vec[0];                                 // years
  duration.mMonths       = vec[1];                                 // months
  duration.mTimeDuration = boost::posix_time::hours(24 * vec[2]) + // days
                           boost::posix_time::hours(vec[3]) +      // hours
                           boost::posix_time::minutes(vec[4]) +    // minutes
                           boost::posix_time::seconds(vec[5]);     // seconds

  if (!duration.isDuration()) {
    spdlog::debug("Input is not valid!");
    return;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void timeDuration(std::string const& isoString, Duration& duration) {
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

void convertIsoDate(std::string& date, boost::posix_time::ptime& time) {
  if (date.find('T') != std::string::npos) {
    if (std::regex_match(date, std::regex("[+\\-][0-9]{2}:?([0-9]{2})?$"))) {
      logger().warn(
          "Time '{}' is not given in UTC but uses an offset. The offset will be ignored!");
    }

    // Remove timezone from date string.
    // For now UTC offsets are not supported and will be ignored.
    date = std::regex_replace(date, std::regex("(Z$|[+\\-][0-9]{2}:?([0-9]{2})?$)"), "");
  }

  date.erase(
      std::remove_if(date.begin(), date.end(), [](unsigned char x) { return std::ispunct(x); }),
      date.end());

  std::size_t pos        = date.find('T');
  std::string dateSubStr = date.substr(0, pos);
  std::string timeSubStr = "T";

  if (pos != std::string::npos) {
    timeSubStr = date.substr(pos);
  }

  if (dateSubStr.size() == 4) {
    // Only year given, append month
    dateSubStr.append("01");
  }
  if (dateSubStr.size() == 6) {
    // Only year and month given, append day
    dateSubStr.append("01");
  }

  dateSubStr.resize(8, '0');
  timeSubStr.resize(7, '0');
  time = boost::posix_time::from_iso_string(dateSubStr + timeSubStr);
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

    TimeInterval             tmp;
    boost::posix_time::ptime start;
    boost::posix_time::ptime end;
    convertIsoDate(startDate, start);

    // Check format of startData
    std::array<std::string, 7> formatParts = {"%Y", "-%m", "-%d", "T%H", ":%M", ":%S", ".%f"};
    std::array<std::string, 7> partLengths = {"4", "2", "2", "2", "2", "2", "1,3"};
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
          tmp.mFormat = "%Y-%m-%dT%H:%M:%S.%fZ";
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

      // If end date is set to currect, select it according to the time format.
      if (endDate == "current") {
        if (tmp.mFormat == "%Y") {
          end = boost::posix_time::ptime(boost::gregorian::date(
              boost::posix_time::microsec_clock::universal_time().date().year(), 1, 1));
        } else if (tmp.mFormat == "%Y-%m") {
          end = boost::posix_time::ptime(boost::gregorian::date(
              boost::posix_time::microsec_clock::universal_time().date().year(),
              boost::posix_time::microsec_clock::universal_time().date().month(), 1));
        } else {
          end = boost::posix_time::microsec_clock::universal_time();
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

bool timeInIntervals(boost::posix_time::ptime& time, std::vector<TimeInterval> const& timeIntervals,
    TimeInterval& foundInterval) {
  // Check each interval whether the given time is inside or not..
  for (auto interval = timeIntervals.rbegin(); interval != timeIntervals.rend(); interval++) {
    // In order to check if there is data for the current time, the length of the WMS interval
    // (duration of one sample) is added to the current interval.
    boost::posix_time::ptime intervalEndTime =
        addDurationToTime(interval->mEndTime, interval->mSampleDuration);

    // Sample time of the current time is inside the time interval.
    if (interval->mStartTime <= time && intervalEndTime > time) {

      // Find the last sample time before the current time.
      if (interval->mSampleDuration.mYears != 0) {
        // Sample rate is in years.

        // Substract years when sample rate is more than one year.
        int year = time.date().year() - ((time.date().year() - interval->mStartTime.date().year()) %
                                            interval->mSampleDuration.mYears);

        // Construct a new time for sample start time.
        time = boost::posix_time::ptime(boost::gregorian::date(year, 1, 1));
      } else if (interval->mSampleDuration.mMonths != 0) {
        // Sample rate is in months.

        // Substract months when sample rate is more than one month.
        int month =
            time.date().month() - ((time.date().month() - interval->mStartTime.date().month()) %
                                      interval->mSampleDuration.mMonths);
        int year = time.date().year();

        // When month is in the previous year.
        if (month > time.date().month()) {
          year -= 1;
        }

        // Construct a new time for sample start time.
        time = boost::posix_time::ptime(boost::gregorian::date(year, month, 1));
      } else {
        // Sample rate is in days or time.

        // Necessary when sample rate is more than 1 day.
        if (interval->mSampleDuration.mTimeDuration.total_seconds() > 0) {
          time -=
              boost::posix_time::seconds((time - interval->mStartTime).total_seconds() %
                                         interval->mSampleDuration.mTimeDuration.total_seconds());
        }
      }

      foundInterval = *interval;

      return true;
    }
  }

  // Time is not in any of the intervals.
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

boost::posix_time::ptime addDurationToTime(
    boost::posix_time::ptime time, Duration const& duration, int multiplier) {

  // Check which unit the interval contatins.
  if (duration.mYears != 0) {
    return time + boost::gregorian::years(multiplier * duration.mYears);
  }
  if (duration.mMonths != 0) {
    return time + boost::gregorian::months(multiplier * duration.mMonths);
  } else {
    return time + duration.mTimeDuration * multiplier;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaXML::TiXmlNode* getElement(
    VistaXML::TiXmlNode* baseElement, std::vector<std::string> const& childPath) {
  VistaXML::TiXmlHandle elementHandle(baseElement);
  for (std::string const& child : childPath) {
    elementHandle = elementHandle.FirstChildElement(child);
  }
  return elementHandle.ToElement();
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
    VistaXML::TiXmlElement const* element, std::string const& attributeName) {
  const std::string* result = element->Attribute(attributeName);
  if (result != nullptr) {
    return *result;
  }
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
std::optional<bool> getAttribute<bool>(
    VistaXML::TiXmlElement const* element, std::string const& attributeName) {
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

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string toStringWithoutTrailing(double value) {
  std::string result = std::to_string(value);
  result.erase(result.find_last_not_of('0') + 1, std::string::npos);
  result.erase(result.find_last_not_of('.') + 1, std::string::npos);
  return result;
};

} // namespace utils

} // namespace csl::ogc