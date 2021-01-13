////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_CONVERSIONS_HPP
#define CS_UTILS_CONVERSIONS_HPP

#include "cs_utils_export.hpp"

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

/// This namespace contains utility functions for converting numbers between different units of
/// measuring. Most of the coordinate system conversion methods are based on the code from the
/// excellent book "3D Engine Design for Virtual Globes" by Patrick Cozzi and Kevin Ring
/// (http://virtualglobebook.com/).
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
  return static_cast<T>(degrees * glm::pi<double>() / 180.0);
}

template <typename T>
T toDegrees(T radians) {
  return static_cast<T>(radians * 180.0 / glm::pi<double>());
}

/// Projects an arbitrary cartesian point to the surface of an origin-centered ellipsoid with the
/// given radii. The projection happens towards the center of the ellipsoid.
CS_UTILS_EXPORT glm::dvec3 scaleToGeocentricSurface(
    glm::dvec3 const& cartesian, glm::dvec3 const& radii);

/// Projects an arbitrary cartesian point to the surface of an origin-centered ellipsoid with the
/// given radii. The projection happens along the surface normal of the ellipsoid. This is
/// potentially a rather expensive operation, as it involes an iterative search for the correct
/// surface normal.
CS_UTILS_EXPORT glm::dvec3 scaleToGeodeticSurface(
    glm::dvec3 const& cartesian, glm::dvec3 const& radii);

/// Computes longitude and geodetic latitude for a given surface point on an origin-centered
/// ellipsoid with the given radii.
CS_UTILS_EXPORT glm::dvec2 surfaceToLngLat(glm::dvec3 const& cartesian, glm::dvec3 const& radii);

/// Computes longitude and geodetic latitude for an arbitrary cartesian point relative to a
/// origin-centered ellipsoid with the given radii. This is potentially a rather expensive
/// operation, as it involes scaleToGeodeticSurface().
CS_UTILS_EXPORT glm::dvec2 cartesianToLngLat(glm::dvec3 const& cartesian, glm::dvec3 const& radii);

/// Same as above, but returns the distance to the surface of the ellipsoid as well.
CS_UTILS_EXPORT glm::dvec3 cartesianToLngLatHeight(
    glm::dvec3 const& cartesian, glm::dvec3 const& radii);

/// Transforms longitude and geodetic latitude and elevation height to cartesian (x,y,z)
/// coordinates for an origin-centered ellipsoid with the given radii. Height is an
/// offset along the geodetic normal of the ellipsoid at lngLat.
CS_UTILS_EXPORT glm::dvec3 toCartesian(
    glm::dvec2 const& lngLat, glm::dvec3 const& radii, double height = 0.0);

/// Returns the geodetic normal vector with unit length at geodetic coordinates (lng, lat) lngLat.
CS_UTILS_EXPORT glm::dvec3 lngLatToNormal(glm::dvec2 const& lngLat);

/// Returns the geodetic normal vector with unit length for a given point on the surface of an
/// origin-centered ellipsoid with the given radii.
CS_UTILS_EXPORT glm::dvec3 surfaceToNormal(glm::dvec3 const& cartesian, glm::dvec3 const& radii);

/// Returns the geodetic normal vector with unit length for an arbitrary cartesian point relative to
/// an origin-centered ellipsoid with the given radii. This is potentially a rather expensive
/// operation, as it involes scaleToGeodeticSurface().
CS_UTILS_EXPORT glm::dvec3 cartesianToNormal(glm::dvec3 const& cartesian, glm::dvec3 const& radii);

/// Time in CosmoScout VR is passed around in different formats.
/// * Strings usually store time in the ISO format YYYY-MM-DDTHH:MM:SS.fffZ. The 'Z' suffix is not
///   really required on the C++ side, as time strings are always considered to be in UTC. This
///   format is also directly convertible to JavaScript Dates, here however the 'Z' is required!
///   Else the Date object will be in your local time zone. So it's a good practice to always append
///   the 'Z'.
/// * boost::posix_time::ptime is used for conversions and is also always in UTC.
/// * SPICE time is stored in doubles representing Barycentric Dynamical Time (TDB, seconds since
///   2000-01-01 12:00:00). Note that this is not the same as UTC seconds since 2000-01-01 12:00:00
///   because TDB considers leap seconds. The conversion methods below take this into account.
namespace time {

/// Converts boost::posix_time::ptime to spice time, which is defined by the Barycentric Dynamical
/// Time. Be aware, that SPICE kernels with leap seconds have to be loaded for this method to work.
/// This means, SolarSystem::init() must have been called before.
CS_UTILS_EXPORT double toSpice(boost::posix_time::ptime const& tIn);

/// Converts a time string to spice time, which is defined by the Barycentric Dynamical Time. The
/// string can be in the format YYYY-MM-DD HH:MM:SS.fff, YYYY-MM-DDTHH:MM:SS.fff, or
/// YYYY-MM-DDTHH:MM:SS.fffZ and is always interpreted as UTC. Be aware, that SPICE kernels with
/// leap seconds have to be loaded for this method to work. This means, SolarSystem::init() must
/// have been called before.
CS_UTILS_EXPORT double toSpice(std::string const& tIn);

/// Converts a time string to boost::posix_time time. The string can be in the format
/// YYYY-MM-DD HH:MM:SS.fff, YYYY-MM-DDTHH:MM:SS.fff, or YYYY-MM-DDTHH:MM:SS.fffZ and is always
/// interpreted as UTC.
CS_UTILS_EXPORT boost::posix_time::ptime toPosix(std::string const& tIn);

/// Converts a Barycentric Dynamical Time to boost::posix_time::ptime. Be aware, that SPICE kernels
/// with leap seconds have to be loaded for this method to work. This means, SolarSystem::init()
/// must have been called before.
CS_UTILS_EXPORT boost::posix_time::ptime toPosix(double tIn);

/// Converts a Barycentric Dynamical Time time to a time string in the format
/// YYYY-MM-DDTHH:MM:SS.fffZ. Be aware, that SPICE kernels with leap seconds have to be loaded for
/// this method to work. This means, SolarSystem::init() must have been called before.
CS_UTILS_EXPORT std::string toString(double tIn);

/// Converts a boost::posix_time::ptime time to a time string in the format
/// YYYY-MM-DDTHH:MM:SS.fffZ.
CS_UTILS_EXPORT std::string toString(boost::posix_time::ptime const& tIn);

} // namespace time

} // namespace cs::utils::convert

#endif // CS_UTILS_CONVERSIONS_HPP
