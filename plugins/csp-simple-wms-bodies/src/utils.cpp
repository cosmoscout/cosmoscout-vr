////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "utils.hpp"

#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-utils/utils.hpp"

namespace csp::simplewmsbodies::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string timeToString(std::string const& format, boost::posix_time::ptime time) {
  std::stringstream sstr;
  auto              facet = new boost::posix_time::time_facet();
  facet->format(format.c_str());
  sstr.imbue(std::locale(std::locale::classic(), facet));
  sstr << time;

  return sstr.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void matchDuration(std::string const& input, std::regex const& re, int& duration) {
  std::smatch match;
  std::regex_search(input, match, re);

  if (match.empty()) {
    spdlog::debug("Pattern does not match!");
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

  duration = 31556926 * vec[0] + // years
             2629744 * vec[1] +  // months
             86400 * vec[2] +    // days
             3600 * vec[3] +     // hours
             60 * vec[4] +       // minutes
             1 * vec[5];         // seconds

  if (duration == 0) {
    spdlog::debug("Input is not valid!");
    return;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void timeDuration(std::string const& isoString, int& duration, std::string& format) {
  std::regex rshort("^((?!T).)*$");
  duration = 0;
  format   = "";

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

  // TODO: base format on the time format in config.
  // Create string format based on interval duration (day / month / year / time).
  if (duration % 86400 == 0 || duration % 2629744 == 0 || duration % 31556926 == 0) {
    format = "%Y-%m-%d";
  } else if (duration % 2629744 == 0) {
    format = "%Y-%m";
  } else if (duration % 31556926 == 0) {
    format = "%Y";
  } else {
    format = "%Y-%m-%dT%H:%MZ";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void convertIsoDate(std::string& date, boost::posix_time::ptime& time) {
  if (date == "current") {
    time = boost::posix_time::microsec_clock::universal_time();
    return;
  }

  date.erase(
      std::remove_if(date.begin(), date.end(), [](unsigned char x) { return std::ispunct(x); }),
      date.end());

  std::string dateSubStr = date.substr(0, date.find("T"));
  std::size_t pos        = date.find("T");
  std::string timeSubStr = "T";

  if (pos != std::string::npos) {
    timeSubStr = date.substr(pos);
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
    std::string       startDate, endDate, duration;
    std::stringstream timeRange_stringstream(timeRange);

    std::getline(timeRange_stringstream, startDate, '/');
    std::getline(timeRange_stringstream, endDate, '/');
    std::getline(timeRange_stringstream, duration, '/');

    TimeInterval             tmp;
    boost::posix_time::ptime start, end;
    convertIsoDate(startDate, start);

    // If there is no end date, just a single timestep.
    if (endDate == "") {
      end                   = start;
      tmp.mIntervalDuration = 0;
      tmp.mFormat           = "%Y-%m-%dT%H:%M:%SZ";
    } else {
      timeDuration(duration, tmp.mIntervalDuration, tmp.mFormat);
      convertIsoDate(endDate, end);
    }

    tmp.mEndTime   = end;
    tmp.mStartTime = start;
    timeIntervals.push_back(tmp);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool timeInIntervals(boost::posix_time::ptime time, std::vector<TimeInterval>& timeIntervals,
    boost::posix_time::time_duration& timeSinceStart, int& intervalDuration, std::string& format) {
  // Check each interval whether the given time is inside or not..
  for (int i = 0; i < timeIntervals.size(); i++) {
    boost::posix_time::time_duration td =
        boost::posix_time::seconds(timeIntervals.at(i).mIntervalDuration);

    if (timeIntervals.at(i).mStartTime <= time && timeIntervals.at(i).mEndTime + td >= time) {
      timeSinceStart   = time - timeIntervals.at(i).mStartTime;
      intervalDuration = timeIntervals.at(i).mIntervalDuration;
      format           = timeIntervals.at(i).mFormat;
      return true;
    }
  }

  // Time is not in any of the intervals.
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::simplewmsbodies::utils