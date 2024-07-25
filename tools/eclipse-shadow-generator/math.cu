////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "math.cuh"

namespace math {

////////////////////////////////////////////////////////////////////////////////////////////////////

double __host__ __device__ angleBetweenVectors(glm::dvec3 const& u, glm::dvec3 const& v) {
  return 2.0 * glm::asin(0.5 * glm::length(u - v));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 __host__ __device__ rotateVector(
    glm::dvec3 const& v, glm::dvec3 const& a, double cosMu) {
  double sinMu = glm::sqrt(1.0 - cosMu * cosMu);
  return v * cosMu + glm::cross(a, v) * sinMu + a * glm::dot(a, v) * (1.0 - cosMu);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double __host__ __device__ getCircleArea(double r) {
  return glm::pi<double>() * r * r;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double __host__ __device__ getCapArea(double r) {
  return 2.0 * glm::pi<double>() * (1.0 - std::cos(r));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double __host__ __device__ getCapIntersection(double rSun, double rOcc, double d) {
  d = std::abs(d);

  if (rSun <= 0.0 || rOcc <= 0.0) {
    return 0.0;
  }

  if (d >= rSun + rOcc) {
    return 0.0;
  }

  if (d <= std::abs(rOcc - rSun)) {
    return getCapArea(glm::min(rSun, rOcc));
  }

  // clang-format off
  return 2.0 * (glm::pi<double>() -
      std::acos((std::cos(d)    - std::cos(rSun) * std::cos(rOcc)) / (std::sin(rSun) * std::sin(rOcc)))
    - std::acos((std::cos(rOcc) - std::cos(d)    * std::cos(rSun)) / (std::sin(d)    * std::sin(rSun))) * std::cos(rSun)
    - std::acos((std::cos(rSun) - std::cos(d)    * std::cos(rOcc)) / (std::sin(d)    * std::sin(rOcc))) * std::cos(rOcc));
  // clang-format on
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double __host__ __device__ getCircleIntersection(double rSun, double rOcc, double d) {
  d = std::abs(d);

  if (rSun <= 0.0 || rOcc <= 0.0) {
    return 0.0;
  }

  if (d >= rSun + rOcc) {
    return 0.0;
  }

  if (d <= std::abs(rOcc - rSun)) {
    return getCircleArea(glm::min(rSun, rOcc));
  }

  double d1 = (rSun * rSun - rOcc * rOcc + d * d) / (2 * d);
  double d2 = d - d1;

  return rSun * rSun * std::acos(d1 / rSun) - d1 * std::sqrt(rSun * rSun - d1 * d1) +
         rOcc * rOcc * std::acos(d2 / rOcc) - d2 * std::sqrt(rOcc * rOcc - d2 * d2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double __host__ __device__ sampleCircleIntersection(
    double rSun, double rOcc, double d, common::LimbDarkening const& limbDarkening) {

  // Sanity checks.
  d = std::abs(d);
  if (rSun <= 0.0 || rOcc <= 0.0) {
    return 0.0;
  }

  // There is no overlapping at all.
  if (d >= rSun + rOcc) {
    return 0.0;
  }

  // The Sun is fully occluded.
  if (d + rSun <= rOcc) {
    return getCircleArea(rSun);
  }

  // We sample a rectangular region which covers the upper half of the intersection area.
  double sampleAreaMinX = d - rOcc;
  double sampleAreaMaxX = glm::min(rSun, d + rOcc);
  double sampleAreaMinY = 0.0;
  double sampleAreaMaxY = glm::min(rSun, rOcc);

  // If both circles are so much apart, we do not have to sample up to
  // glm::min(rSun, rOcc) vertically. We get the required sample height with Heron's
  // formula.
  if (d * d + std::pow(glm::min(rSun, rOcc), 2.0) > std::pow(glm::max(rSun, rOcc), 2.0)) {
    double a       = rSun;
    double b       = rOcc;
    double c       = d;
    double s       = 0.5 * (a + b + c);
    sampleAreaMaxY = 2.0 * std::sqrt(s * (s - a) * (s - b) * (s - c)) / c;
  }

  const int32_t xSamples = 512;
  const int32_t ySamples = xSamples / 2;
  double        area     = 0.0;

  glm::dvec2 samplePos;

  for (int32_t y(0); y < ySamples; ++y) {
    samplePos.y = (1.0 * y + 0.5) / ySamples;
    samplePos.y = sampleAreaMinY + samplePos.y * (sampleAreaMaxY - sampleAreaMinY);

    for (int32_t x(0); x < xSamples; ++x) {
      samplePos.x = (1.0 * x + 0.5) / xSamples;
      samplePos.x = sampleAreaMinX + samplePos.x * (sampleAreaMaxX - sampleAreaMinX);

      if (glm::distance2(glm::dvec2(0.0, 0.0), samplePos) < rSun * rSun &&
          glm::distance2(glm::dvec2(d, 0.0), samplePos) < rOcc * rOcc) {
        area += (sampleAreaMaxX - sampleAreaMinX) / xSamples * (sampleAreaMaxY - sampleAreaMinY) /
                ySamples * limbDarkening.get(glm::length(samplePos) / rSun);
      }
    }
  }

  // We sampled only half the area.
  return area * 2.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void __host__ __device__ mapPixelToRadii(glm::ivec2 const& pixel, uint32_t resolution,
    common::Mapping const& mapping, double& radiusOcc, double& distance) {

  double x = glm::pow((1.0 * pixel.x + 0.5) / resolution, mapping.mExponent);
  double y = 1.0 - glm::pow(1.0 - (1.0 * pixel.y + 0.5) / resolution, mapping.mExponent);

  double radiusSun = 1.0;
  radiusOcc        = radiusSun / x - radiusSun;

  double minDistance = mapping.mIncludeUmbra ? 0.0 : glm::max(radiusOcc - radiusSun, 0.0);
  double maxDistance = radiusOcc + radiusSun;

  distance = minDistance + y * (maxDistance - minDistance);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t __host__ __device__ mapPixelToAngles(glm::ivec2 const& pixel, uint32_t resolution,
    common::Mapping const& mapping, common::Geometry const& geometry, double& phiOcc,
    double& phiSun, double& delta) {

  double radiusOcc, distance;
  mapPixelToRadii(pixel, resolution, mapping, radiusOcc, distance);

  phiSun = glm::asin(geometry.mRadiusSun / geometry.mSunOccDist);
  phiOcc = radiusOcc * phiSun;
  delta  = distance * phiSun;

  double   error      = 1.0;
  uint32_t iterations = 0;

  while (error > 0.0001 && ++iterations < 100) {
    double occDist = geometry.mRadiusOcc / glm::sin(phiOcc);
    double sunDist = occDist * glm::cos(delta) +
                     glm::sqrt(occDist * occDist * glm::cos(delta) * glm::cos(delta) -
                               occDist * occDist + geometry.mSunOccDist * geometry.mSunOccDist);

    double newPhiSun = glm::asin(geometry.mRadiusSun / sunDist);
    double newPhiOcc = radiusOcc * phiSun;
    double newDelta  = distance * phiSun;

    error = glm::max(glm::abs(phiSun - newPhiSun) / phiSun,
        glm::max(glm::abs(phiOcc - newPhiOcc) / phiOcc, glm::abs(delta - newDelta) / delta));

    phiSun = newPhiSun;
    phiOcc = newPhiOcc;
    delta  = newDelta;
  }

  return iterations;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace math
