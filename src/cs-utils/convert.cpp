////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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

double toSpice(boost::posix_time::ptime const& tIn) {

  auto const startYear       = 2000;
  auto const noon            = 12;
  auto const secondsToMillis = 1000.0;

  auto j2000 = boost::posix_time::ptime(
      boost::gregorian::date(startYear, 1, 1), boost::posix_time::hours(noon));

  double dTime = (tIn - j2000).total_milliseconds() / secondsToMillis;

  // Incorporate delta between ET and UTC.
  double ETUTCDelta = 0.0;
  deltet_c(dTime, "UTC", &ETUTCDelta);

  return dTime + ETUTCDelta;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double toSpice(std::string const& tIn) {
  try {
    return toSpice(toPosix(tIn));
  } catch (std::exception& e) { logger().error("Failed to convert time: {}", e.what()); }

  return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

boost::posix_time::ptime toPosix(std::string const& tIn) {

  // We need at least a length of 19 chracters
  if (tIn.length() < 19) {
    logger().error("Failed to convert '{}' to boost::posix_time::ptime!", tIn);
    return boost::posix_time::ptime();
  }

  // Remove potential Z in YYYY-MM-DDTHH:MM:SS.fffZ
  auto copy = tIn;
  if (copy.back() == 'Z') {
    copy.back() = '0';
  }

  // Remove potential T in YYYY-MM-DDTHH:MM:SS.fff
  copy[10] = ' ';

  try {
    // Let boost do the parsing
    return boost::posix_time::time_from_string(copy);

  } catch (std::exception& e) {
    logger().error("Failed to convert '{}' to boost::posix_time::ptime: {}!", tIn, e.what());
  }

  return boost::posix_time::ptime();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

boost::posix_time::ptime toPosix(double tIn) {
  auto const startYear       = 2000;
  auto const noon            = 12;
  auto const secondsToMillis = 1000;

  // Incorporate delta between ET and UTC.
  double ETUTCDelta = 0.0;
  deltet_c(tIn, "ET", &ETUTCDelta);

  return boost::posix_time::ptime(boost::gregorian::date(startYear, 1, 1),
      boost::posix_time::hours(noon) + boost::posix_time::milliseconds(static_cast<int64_t>(
                                           (tIn - ETUTCDelta) * secondsToMillis)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string toString(double tIn) {
  return toString(toPosix(tIn));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string toString(boost::posix_time::ptime const& tIn) {
  return boost::posix_time::to_iso_extended_string(tIn).substr(0, 23) + "Z";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace time

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils::convert
