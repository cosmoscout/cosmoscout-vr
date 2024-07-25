////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef MATH_HPP
#define MATH_HPP

#include <algorithm>

#define GLM_FORCE_CTOR_INIT
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>

#include "LimbDarkening.cuh"
#include "common.hpp"

namespace math {

// Using acos is not very stable for small angles. This function uses asin to compute the angle
// between two vectors in a more stable way.
double __host__ __device__ angleBetweenVectors(glm::dvec3 const& u, glm::dvec3 const& v);

// Rotates vector v around axis a using Rodrigues' rotation formula.
glm::dvec3 __host__ __device__ rotateVector(glm::dvec3 const& v, glm::dvec3 const& a, double cosMu);

// Returns the surface area of a circle.
double __host__ __device__ getCircleArea(double r);

// Returns the surface area of a spherical cap on a unit sphere.
double __host__ __device__ getCapArea(double r);

// Returns the intersection area of two spherical caps with radii rSun and rOcc whose center points
// are distance d away from each other. All values are given as angles on the unit sphere.
double __host__ __device__ getCapIntersection(double rSun, double rOcc, double d);

// Returns the intersection area of two circles with radii rSun and rOcc whose center points are
// distance d away from each other.
double __host__ __device__ getCircleIntersection(double rSun, double rOcc, double d);

// Same as above, but the intersection area is computed by sampling. This is less precise but
// allows for incorporating limb darkening.
double __host__ __device__ sampleCircleIntersection(
    double rSun, double rOcc, double d, common::LimbDarkening const& limbDarkening);

// Maps a pixel position in the shadow map to a corresponding radius of a 2D projection of the
// occluder, and a distance between the occluder and the Sun. Both values are scaled in such a way
// that the radius of the Solar disc is 1.0. This is usually sufficient for computing the visible
// fraction of the Sun.
// To get the actual angular radii and angular distances, more information on the involved geometry
// is needed. For this, use the function below.
void __host__ __device__ mapPixelToRadii(glm::ivec2 const& pixel, uint32_t resolution,
    common::Mapping const& mapping, double& radiusOcc, double& distance);

// To reconstruct the actual geometry of the involved bodies for a given location in the shadow map,
// we need additional information like the real-world radii of the Sun and the occluder, as well as
// their distance to each other.
//
// It happens that this reconstruction is a non-trivial problem which cannot be solved analytically.
// Even if there is only a single solution, it seems to be impossible to find it without numerical
// methods.
//
// Our approach is as follow:
//  1. The largest possible angular radius of the Sun from any position in the occluder's shadow
//     is its angular radius when observed from the occluder's position. Therefore, we scale all
//     values returned by mapPixelToRadii this value. Most likely, the Sun will appear even smaller
//     as our searched point is further away from the occluder, but we take this as an initial
//     guess.
//  2. We now use the real radius of the occluder, the average distance between Sun and occluder,
//     and the angular distance between the centers of Sun and occluder to compute the resulting
//     distance to the Sun from the searched point. This will be farther away than the initial
//     guess.
//  3. Using the real radius of the Sun, we can now compute again how large the Sun will appear
//     from the searched point. This will be smaller than the initial guess. We again scale all
//     angles to this value and repeat the process until we converge.
//
// The method returns the number of iterations needed to converge.
uint32_t __host__ __device__ mapPixelToAngles(glm::ivec2 const& pixel, uint32_t resolution,
    common::Mapping const& mapping, common::Geometry const& geometry, double& phiOcc,
    double& phiSun, double& delta);

} // namespace math

#endif // MATH_HPP