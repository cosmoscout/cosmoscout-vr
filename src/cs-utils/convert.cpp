////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "convert.hpp"

#include "logger.hpp"

#include <cmath>
#include <cspice/SpiceUsr.h>
#include <glm/gtc/type_ptr.hpp>

namespace cs::utils::convert {

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 scaleToGeocentricSurface(glm::dvec3 const& cartesian, glm::dvec3 const& radii) {
  double beta = 1.0 / std::sqrt(glm::dot(cartesian * cartesian, 1.0 / (radii * radii)));
  return cartesian * beta;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 scaleToGeodeticSurface(glm::dvec3 const& cartesian, glm::dvec3 const& radii) {

  auto radii2        = radii * radii;
  auto radii4        = radii2 * radii2;
  auto oneOverRadii2 = 1.0 / radii2;
  auto cartesian2    = cartesian * cartesian;

  double beta  = 1.0 / std::sqrt(glm::dot(cartesian2, oneOverRadii2));
  double n     = glm::length(beta * cartesian * oneOverRadii2);
  double alpha = (1.0 - beta) * (glm::length(cartesian) / n);
  double s     = 0.0;
  double dSdA  = 1.0;

  glm::dvec3 d;

  do {
    alpha -= (s / dSdA);

    d    = glm::dvec3(1.0) + (alpha * oneOverRadii2);
    s    = glm::dot(cartesian2, 1.0 / (radii2 * d * d)) - 1.0;
    dSdA = glm::dot(cartesian2, 1.0 / (radii4 * d * d * d)) * -2.0;

  } while (std::abs(s) > 1e-10);

  return cartesian / d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec2 surfaceToLngLat(glm::dvec3 const& cartesian, glm::dvec3 const& radii) {
  auto geodeticNormal = surfaceToNormal(cartesian, radii);
  return glm::dvec2(std::atan2(geodeticNormal.x, geodeticNormal.z), std::asin(geodeticNormal.y));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec2 cartesianToLngLat(glm::dvec3 const& cartesian, glm::dvec3 const& radii) {
  auto surfacePoint = scaleToGeodeticSurface(cartesian, radii);
  return surfaceToLngLat(surfacePoint, radii);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 cartesianToLngLatHeight(glm::dvec3 const& cartesian, glm::dvec3 const& radii) {
  auto   surfacePoint = scaleToGeodeticSurface(cartesian, radii);
  auto   dir          = cartesian - surfacePoint;
  double height       = std::copysign(1.0, glm::dot(dir, cartesian)) * glm::length(dir);

  return glm::dvec3(surfaceToLngLat(surfacePoint, radii), height);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 toCartesian(glm::dvec2 const& lngLat, glm::dvec3 const& radii, double height) {
  auto normal  = lngLatToNormal(lngLat);
  auto normal2 = normal * normal;
  auto radii2  = radii * radii;
  auto point   = (radii2 * normal) / std::sqrt(glm::dot(radii2, normal2));

  return point + normal * height;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 lngLatToNormal(glm::dvec2 const& lngLat) {
  return glm::dvec3(std::cos(lngLat.y) * std::sin(lngLat.x), std::sin(lngLat.y),
      std::cos(lngLat.y) * std::cos(lngLat.x));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 surfaceToNormal(glm::dvec3 const& cartesian, glm::dvec3 const& radii) {
  auto radii2        = radii * radii;
  auto oneOverRadii2 = 1.0 / radii2;
  return glm::normalize(cartesian * oneOverRadii2);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 cartesianToNormal(glm::dvec3 const& cartesian, glm::dvec3 const& radii) {
  return surfaceToNormal(scaleToGeodeticSurface(cartesian, radii), radii);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace time {

////////////////////////////////////////////////////////////////////////////////////////////////////

double toSpice(std::chrono::utc_clock::time_point const& tIn) {
  using namespace std::chrono_literals;

  auto j2000 = toUTC(2000y / 1 / 1, 12h);

  // Calculate time difference in seconds
  auto diff  = tIn - j2000;
  auto dTime = std::chrono::duration_cast<std::chrono::duration<double>>(diff).count();

  // Incorporate delta between ET and UTC.
  double ETUTCDelta = 0.0;
  deltet_c(dTime, "UTC", &ETUTCDelta);

  return dTime + ETUTCDelta;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double toSpice(std::string const& tIn) {
  try {
    return toSpice(toUTC(tIn));
  } catch (std::exception& e) { logger().error("Failed to convert time: {}", e.what()); }

  return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::chrono::utc_clock::time_point toUTC(const std::string& tIn) {
  // We need at least a length of 19 characters
  if (tIn.length() < 19) {
    logger().error("Failed to convert '{}' to time_point!", tIn);
    return std::chrono::utc_clock::time_point{};
  }

  try {
    // Create a copy for potential modifications
    std::string copy = tIn;

    // Remove potential Z in YYYY-MM-DDTHH:MM:SS.fffZ
    if (copy.back() == 'Z') {
      copy.pop_back();
    }

    // Replace T with space if present at position 10
    if (copy.length() > 10 && copy[10] == 'T') {
      copy[10] = ' ';
    }

    // Parse date and time using C++20 chrono
    std::istringstream          ss(copy);
    std::chrono::year_month_day date;

    int  year, month, day;
    char delimiter;
    ss >> year >> delimiter >> month >> delimiter >> day;

    if (ss.fail()) {
      throw std::runtime_error("Failed to parse date part");
    }

    date = std::chrono::year{year} / static_cast<std::chrono::month>(month) /
           static_cast<std::chrono::day>(day);

    if (!date.ok()) {
      throw std::runtime_error("Invalid date");
    }

    int hour, minute, second;
    ss >> hour >> delimiter >> minute >> delimiter >> second;

    if (ss.fail()) {
      throw std::runtime_error("Failed to parse time part");
    }

    // Handle potential milliseconds
    int milliseconds = 0;
    if (ss.peek() == '.') {
      ss.ignore(); // Skip the dot
      ss >> milliseconds;
    }

    auto time = std::chrono::hours{hour} + std::chrono::minutes{minute} +
                std::chrono::seconds{second} + std::chrono::milliseconds{milliseconds};

    // Combine date and time
    return toUTC(date, time);
  } catch (const std::exception& e) {
    logger().error("Failed to convert '{}' to time_point: {}!", tIn, e.what());
  }

  return std::chrono::utc_clock::time_point{};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::chrono::utc_clock::time_point toUTC(double tIn) {
  using namespace std::chrono_literals;

  // Incorporate delta between ET and UTC.
  double ETUTCDelta = 0.0;
  deltet_c(tIn, "ET", &ETUTCDelta);

  // Create J2000 reference time (January 1, 2000 at 12:00:00 UTC)
  auto j2000 = toUTC(2000y / 1 / 1, 12h);

  // Add the milliseconds offset
  auto seconds = std::chrono::duration<double>(tIn - ETUTCDelta);

  // Return the computed time point
  return j2000 + std::chrono::duration_cast<std::chrono::utc_clock::duration>(seconds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::chrono::utc_clock::time_point toUTC(
    std::chrono::year_month_day const& ymd, std::chrono::utc_clock::duration const& hms) {
  return std::chrono::utc_clock::from_sys(std::chrono::sys_days{ymd} + hms);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string toString(double tIn) {
  return toString(toUTC(tIn));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string toString(std::chrono::utc_clock::time_point const& tIn) {
  using namespace std::chrono_literals;

  // This method is mainly used by creating a JavaScript date object, which doesn't support leap
  // seconds. The formatter below does support them and would generate a 60 in the seconds field,
  // which would not work with JavaScript's date parsing. We handle this here by subtracting a
  // second during a leap second and displaying 59 seconds, instead of 60 seconds.
  bool isLeapSecond = std::chrono::get_leap_second_info(tIn).is_leap_second;

  auto time =
      std::chrono::time_point_cast<std::chrono::milliseconds>(tIn - (isLeapSecond ? 1s : 0s));

  // Format the time point as ISO 8601 string with milliseconds precision
  // and append 'Z' to indicate UTC time zone
  return std::format("{:%Y-%m-%dT%H:%M:%S}Z", time);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace time

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils::convert
