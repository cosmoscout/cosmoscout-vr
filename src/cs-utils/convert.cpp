////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "convert.hpp"

#include <cspice/SpiceUsr.h>
#include <glm/gtc/type_ptr.hpp>
#include <spdlog/spdlog.h>

namespace cs::utils::convert {

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 toLngLatHeight(glm::dvec3 const& cartesian, double radiusE, double radiusP) {
  // Calculate longitude.
  double longitude = 0.0;

  if (cartesian.z != 0.0) {
    longitude = std::atan(cartesian.x / cartesian.z);

    if (cartesian.z < 0 && cartesian.x < 0) {
      longitude -= glm::pi<double>();
    }

    if (cartesian.z < 0 && cartesian.x >= 0) {
      longitude += glm::pi<double>();
    }
  } else if (cartesian.x == 0) {
    longitude = 0.0;
  } else if (cartesian.x < 0) {
    longitude = -glm::pi<double>() * 0.5;
  } else {
    longitude = glm::pi<double>() * 0.5;
  }

  // Geocentric latitude of the input point.
  double latitude = std::asin(cartesian.y / glm::length(cartesian));

  // This latitude corresponds to the geocentric latitude of the intersection point of the ellipsoid
  // with the line between it's center and the cartesian input location. We can calculate the
  // cartesian position of this intersection.
  latitude = geocentricToGeodetic(latitude, radiusE, radiusP);
  glm::dvec2 lngLatIntersection(longitude, latitude);
  glm::dvec3 intersection = toCartesian(lngLatIntersection, radiusE, radiusP);

  // We assume the distance between this intersection and the input point to be the height. This is
  // actually wrong and needs to be fixed! (issue #28)
  double height = glm::length(cartesian) - glm::length(intersection);

  // We return here the geodetic latitude of the surface point which has the same geocentric
  // latitude as the input point - this is wrong too, and should be fixed (issue #28)
  return glm::dvec3(lngLatIntersection, height);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 toCartesian(glm::dvec2 const& lngLat, double radiusE, double radiusP, double height) {
  glm::dvec2 lngLatGeoCentric(lngLat.x, geodeticToGeocentric(lngLat.y, radiusE, radiusP));

  glm::dvec2 c = glm::cos(lngLatGeoCentric);
  glm::dvec2 s = glm::sin(lngLatGeoCentric);

  glm::dvec3 cartesian;
  cartesian.x = radiusE * c[1] * s[0];
  cartesian.y = radiusP * s[1];
  cartesian.z = radiusE * c[1] * c[0];

  if (height != 0.0) {
    cartesian += height * lngLatToNormal(lngLat, radiusE, radiusP);
  }

  return cartesian;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double geocentricToGeodetic(double lat, double radiusE, double radiusP) {
  double f = (radiusE - radiusP) / radiusE;
  return std::atan(std::tan(lat) / std::pow(1.0 - f, 2.0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double geocentricToParametric(double lat, double radiusE, double radiusP) {
  double f = (radiusE - radiusP) / radiusE;
  return std::atan(std::tan(lat) / (1.0 - f));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double geodeticToGeocentric(double lat, double radiusE, double radiusP) {
  double f = (radiusE - radiusP) / radiusE;
  return std::atan(std::tan(lat) * std::pow(1.0 - f, 2.0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double geodeticToParametric(double lat, double radiusE, double radiusP) {
  double f = (radiusE - radiusP) / radiusE;
  return std::atan(std::tan(lat) * (1.0 - f));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double parametricToGeocentric(double lat, double radiusE, double radiusP) {
  double f = (radiusE - radiusP) / radiusE;
  return std::atan(std::tan(lat) * (1.0 - f));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double parametricToGeodetic(double lat, double radiusE, double radiusP) {
  double f = (radiusE - radiusP) / radiusE;
  return std::atan(std::tan(lat) / (1.0 - f));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 lngLatToNormal(glm::dvec2 const& lngLat, double radiusE, double radiusP) {
  glm::dvec3 cart = toCartesian(lngLat, radiusE, radiusP);
  glm::dvec3 n    = cart / glm::dvec3(radiusE * radiusE, radiusP * radiusP, radiusE * radiusE);
  return glm::normalize(n);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec2 normalToLngLat(glm::dvec3 const& normal, double radiusE, double radiusP) {
  glm::dvec3 cart = glm::normalize(normal * glm::dvec3(radiusE, radiusP, radiusE));

  glm::dvec2 lngLat;
  lngLat.y = std::asin(cart.y);

  if (cart.z != 0.0) {
    lngLat.x = std::atan(cart.x / cart.z);

    if (cart.z < 0 && cart.x < 0) {
      lngLat.x -= glm::pi<double>();
    }

    if (cart.z < 0 && cart.x >= 0) {
      lngLat.x += glm::pi<double>();
    }
  } else if (cart.x == 0) {
    lngLat.x = 0.0;
  } else if (cart.x < 0) {
    lngLat.x = -glm::pi<double>() * 0.5;
  } else {
    lngLat.x = glm::pi<double>() * 0.5;
  }

  lngLat.y = geocentricToGeodetic(lngLat.y, radiusE, radiusP);

  return lngLat;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double toSpiceTime(boost::posix_time::ptime const& tIn) {
  double dTime;
  str2et_c(boost::posix_time::to_simple_string(tIn).c_str(), &dTime);
  return dTime;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double toSpiceTime(std::string const& tIn) {
  try {
    return toSpiceTime(boost::posix_time::time_from_string(tIn));
  } catch (std::exception& e) { spdlog::error("Failed to convert time: {}", e.what()); }

  return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

boost::posix_time::ptime toBoostTime(double tIn) {
  return boost::posix_time::ptime(boost::gregorian::date(2000, 1, 1),
      boost::posix_time::hours(12) +
          boost::posix_time::milliseconds(static_cast<int64_t>(tIn * 1000)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils::convert
