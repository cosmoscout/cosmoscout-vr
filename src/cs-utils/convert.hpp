////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_CONVERSIONS_HPP
#define CS_UTILS_CONVERSIONS_HPP

#include "cs_utils_export.hpp"

#include "../cs-core/Settings.hpp"

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

/// This namespace contains utility functions for converting numbers between different units of
/// measuring.
namespace cs::utils::convert {

template <typename T>
T lightyearsToMeters(T lightyears) {
  return lightyears * 9460730472580800.0;
}

template <typename T>
T metersToLightyears(T meters) {
  return meters / 9460730472580800.0;
}

/// Converts AU to meters. One AU is equivalent to the average distance between the Earth and the
///// Sun.
template <typename T>
T astronomicalUnitsToMeters(T astronomicalUnits) {
  return astronomicalUnits * 149597870700.0;
}

/// Converts meters to AU. One AU is equivalent to the average distance between the Earth and the
/// Sun.
template <typename T>
T metersToAstronomicalUnits(T meters) {
  return meters / 149597870700.0;
}

template <typename T>
T toRadians(T degrees) {
  return degrees * glm::pi<double>() / 180.0;
}

template <typename T>
T toDegrees(T radians) {
  return radians * 180.0 / glm::pi<double>();
}

/// Transform cartesian (x,y,z) coordinates to geodetic (lng, lat, height above surface)
/// coordinates.
CS_UTILS_EXPORT glm::dvec3 toLngLatHeight(
    glm::dvec3 const& cartesian, double radiusE, double radiusP);

/// Transform geodetic coordinates (lng, lat) LngLat and elevation height to cartesian (x,y,z)
/// coordinates for an ellipsoid of equatorial radius radiusE and polar radius radiusP. Height is an
/// offset along the normal of the ellipsoid at (lng, lat).
CS_UTILS_EXPORT glm::dvec3 toCartesian(
    glm::dvec2 const& lngLat, double radiusE, double radiusP, double height = 0.0);

/// Convert latitudes.
CS_UTILS_EXPORT double geocentricToGeodetic(double lat, double radiusE, double radiusP);
CS_UTILS_EXPORT double geocentricToParametric(double lat, double radiusE, double radiusP);
CS_UTILS_EXPORT double geodeticToGeocentric(double lat, double radiusE, double radiusP);
CS_UTILS_EXPORT double geodeticToParametric(double lat, double radiusE, double radiusP);
CS_UTILS_EXPORT double parametricToGeocentric(double lat, double radiusE, double radiusP);
CS_UTILS_EXPORT double parametricToGeodetic(double lat, double radiusE, double radiusP);

/// Returns the normal vector (unit length) to the ellipsoid with equatorial radius radiusE and
/// polar radius radiusP at geodetic coordinates (lng, lat) LngLat.
CS_UTILS_EXPORT glm::dvec3 lngLatToNormal(glm::dvec2 const& lngLat, double radiusE, double radiusP);

/// Returns the geodetic coordinates (lng, lat) for a given normal vector.
CS_UTILS_EXPORT glm::dvec2 normalToLngLat(glm::dvec3 const& normal, double radiusE, double radiusP);

/// Parse the time of existence from the settings section of an anchor.
CS_UTILS_EXPORT std::pair<double, double> getExistenceFromSettings(std::pair<std::string, core::Settings::Anchor> anchor);

/// Convert boost::posix_time::ptime to spice time, which is defined by the
/// Barycentric Dynamical Time.
CS_UTILS_EXPORT double toSpiceTime(boost::posix_time::ptime const& tIn);

/// Convert a time string to spice time, which is defined by the Barycentric Dynamical Time.
CS_UTILS_EXPORT double toSpiceTime(std::string const& tIn);

/// Convert time in seconds since 2000-01-01 12:00:00.000 to boost::posix_time::ptime. Be
/// aware, that fractional seconds will be truncated. //DocTODO
CS_UTILS_EXPORT boost::posix_time::ptime toBoostTime(double tIn);

} // namespace cs::utils::convert

#endif // CS_UTILS_CONVERSIONS_HPP
