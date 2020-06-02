////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_UTIL_HPP
#define CSP_LOD_BODIES_UTIL_HPP

#define _USE_MATH_DEFINES // Use Math.h defines like M_PI
#include <cmath>          // C++ Math
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

class VistaTransformNode;
class VistaOpenGLNode;

namespace csp::lodbodies {

class VistaPlanet;
class TileNode;

/// Defines the Sample Precision
enum class HeightSamplePrecision {
  eCoarse = 1, ///< Use the Base Patches (LOD 0) only.
  eActual = 2, ///< Use the already loaded Patches.
  eFine   = 3  ///< Use the highest LOD available in database.
};

namespace utils {

/// Retrieve the Planets Height at a specific lat / long position.
/// @param planet    VistaPlanet to get the Height from
/// @param precision Defines the Height Sample Precision
/// @param position  Where X is longitude (-PI - PI) West to East and  Y is geocentric latitude
///                  (PI/2 - -PI/2) North to South
double getHeight(
    VistaPlanet const* planet, HeightSamplePrecision precision, glm::dvec2 const& lngLat);

/// Intersects a ray with the height field of a VistaPlanet. The Ray is defined by a position
/// and orientation.
/// @param planet VistaPlanet to be intersected
/// @param rayPos Ray position in world space
/// @param rayDir Ray direction in world space
/// @param pos    Gives (first) intersection in cartesian
bool intersectPlanet(
    VistaPlanet const* planet, glm::dvec3 rayOrigin, glm::dvec3 rayDir, glm::dvec3& pos);

/// Retrieve entry and exit distance along a ray on the bbox of a tile node. The ray parameters must
/// be transformed into the planet coordinate system before!
bool intersectTileBounds(TileNode const* tileNode, VistaPlanet const* planet,
    glm::dvec4 const& origin, glm::dvec4 const& direction, double& minDist, double& maxDist);

} // namespace utils

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_UTIL_HPP
