////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MATH_HPP
#define MATH_HPP

#include <algorithm>

#define GLM_FORCE_CTOR_INIT
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>

#include "LimbDarkening.cuh"

namespace math {

/// Returns the surface area of a circle.
double __host__ __device__ getCircleArea(double r);

/// Returns the surface area of a spherical cap on a unit sphere.
double __host__ __device__ getCapArea(double r);

/// Returns the intersection area of two spherical caps with radii rSun and rOcc whose center points
/// are distance d away from each other. All values are given as angles on the unit sphere.
double __host__ __device__ getCapIntersection(double rSun, double rOcc, double d);

/// Returns the intersection area of two circles with radii rSun and rOcc whose center points are
/// distance d away from each other.
double __host__ __device__ getCircleIntersection(double rSun, double rOcc, double d);

/// Same as above, but the intersection area is computed by sampling. This is less precise but
/// allows for incorporating limb darkening.
double __host__ __device__ sampleCircleIntersection(
    double rSun, double rOcc, double d, LimbDarkening const& limbDarkening);

/// Maps a pixel position in the shadow map to the three angles phiSun, phiOcc and delta. phiSun is
/// assumed to be 1.0, phiOcc is returned as first component, delta as second component.
glm::dvec2 __host__ __device__ mapPixelToAngles(
    glm::ivec2 const& pixel, uint32_t resolution, double exponent, bool includeUmbra);

/// Maps the three angles phiSun, phiOcc and delta to a pixel position in the shadow map to. phiSun
/// is assumed to be 1.0, phiOcc should be given as first component, delta as second component of
/// "angled".
glm::ivec2 __host__ __device__ mapAnglesToPixel(
    glm::dvec2 const& angles, uint32_t resolution, double exponent, bool includeUmbra);
} // namespace math

#endif // MATH_HPP