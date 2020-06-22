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

glm::dvec3 toLngLatHeight(glm::dvec3 const& cartesian, glm::dvec3 const& radii) {

  auto   surfacePoint   = scaleToGeodeticSurface(cartesian, radii);
  auto   dir            = cartesian - surfacePoint;
  double height         = std::copysign(1.0, glm::dot(dir, cartesian)) * glm::length(dir);
  auto   geodeticNormal = cartesianToNormal(surfacePoint, radii);

  return glm::dvec3(
      std::atan2(geodeticNormal.x, geodeticNormal.z), std::asin(geodeticNormal.y), height);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 scaleToGeocentricSurface(glm::dvec3 const& cartesian, glm::dvec3 const& radii) {
  double beta = 1.0 / std::sqrt(cartesian.x * cartesian.x / (radii.x * radii.x) +
                                cartesian.y * cartesian.y / (radii.y * radii.y) +
                                cartesian.z * cartesian.z / (radii.z * radii.z));
  return cartesian * beta;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 scaleToGeodeticSurface(glm::dvec3 const& cartesian, glm::dvec3 const& radii) {

  auto radiiSquared        = radii * radii;
  auto oneOverRadiiSquared = 1.0 / radiiSquared;
  auto radiiToTheFourth    = radiiSquared * radiiSquared;

  double beta = 1.0 / std::sqrt((cartesian.x * cartesian.x) * oneOverRadiiSquared.x +
                                (cartesian.y * cartesian.y) * oneOverRadiiSquared.y +
                                (cartesian.z * cartesian.z) * oneOverRadiiSquared.z);

  double n = glm::length(beta * cartesian * oneOverRadiiSquared);

  double alpha = (1.0 - beta) * (glm::length(cartesian) / n);

  double x2 = cartesian.x * cartesian.x;
  double y2 = cartesian.y * cartesian.y;
  double z2 = cartesian.z * cartesian.z;

  double da = 0.0;
  double db = 0.0;
  double dc = 0.0;

  double s    = 0.0;
  double dSdA = 1.0;

  do {
    alpha -= (s / dSdA);

    da = 1.0 + (alpha * oneOverRadiiSquared.x);
    db = 1.0 + (alpha * oneOverRadiiSquared.y);
    dc = 1.0 + (alpha * oneOverRadiiSquared.z);

    double da2 = da * da;
    double db2 = db * db;
    double dc2 = dc * dc;

    double da3 = da * da2;
    double db3 = db * db2;
    double dc3 = dc * dc2;

    s = x2 / (radiiSquared.x * da2) + y2 / (radiiSquared.y * db2) + z2 / (radiiSquared.z * dc2) -
        1.0;

    dSdA = -2.0 * (x2 / (radiiToTheFourth.x * da3) + y2 / (radiiToTheFourth.y * db3) +
                      z2 / (radiiToTheFourth.z * dc3));

  } while (std::abs(s) > 1e-10);

  return glm::dvec3(cartesian.x / da, cartesian.y / db, cartesian.z / dc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 toCartesian(glm::dvec2 const& lngLat, glm::dvec3 const& radii, double height) {

  auto normal = lngLatToNormal(lngLat, radii);
  auto rX2    = radii.x * radii.x;
  auto rY2    = radii.y * radii.y;
  auto rZ2    = radii.z * radii.z;

  double gamma =
      std::sqrt(rX2 * normal.x * normal.x + rY2 * normal.y * normal.y + rZ2 * normal.z * normal.z);

  auto point = glm::dvec3(rX2 * normal.x, rY2 * normal.y, rZ2 * normal.z) / gamma;

  return point + normal * height;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 lngLatToNormal(glm::dvec2 const& lngLat, glm::dvec3 const& radii) {
  return glm::dvec3(std::cos(lngLat.y) * std::sin(lngLat.x), std::sin(lngLat.y),
      std::cos(lngLat.y) * std::cos(lngLat.x));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 surfacePosToNormal(glm::dvec3 const& surfacePos, glm::dvec3 const& radii) {
  auto radiiSquared        = radii * radii;
  auto oneOverRadiiSquared = 1.0 / radiiSquared;
  return glm::normalize(surfacePos * oneOverRadiiSquared);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 cartesianToNormal(glm::dvec3 const& cartesian, glm::dvec3 const& radii) {
  return surfacePosToNormal(scaleToGeodeticSurface(cartesian, radii), radii);
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
