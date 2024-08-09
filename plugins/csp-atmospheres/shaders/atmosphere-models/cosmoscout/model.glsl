////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#version 330

// constants
const float PI = 3.141592653589793;

// uniforms
uniform float uSunIlluminance;

// This atmospheric model uses a pretty basic implementation of single scattering. It requires no
// preprocessing. For each pixel, a primary ray is cast through the atmosphere and at specific
// sample positions, secondary rays are cast towards the Sun. The sun light is attenuated along the
// secondary rays and accumulated along the primary ray.
// Scroll to the bottom of this file to see the public API of this shader.

// -------------------------------------------------------------------------------- internal methods

// The Cornette-Shanks phase function returns the probability of scattering based on the cosine
// between in and out direction (c) and the anisotropy (g). It converges to Rayleigh scattering
// for small values of g.
//
//            3 * (1 - g*g)               1 + c*c
// phase = -------------------- * -----------------------
//          8 * PI * (2 + g*g)     (1 + g*g - 2*g*c)^1.5
//
float _getPhase(float cosine, float anisotropy) {
  float anisotropy2 = anisotropy * anisotropy;
  float cosine2     = cosine * cosine;

  float a = (1.0 - anisotropy2) * (1.0 + cosine2);
  float b = 1.0 + anisotropy2 - 2.0 * anisotropy * cosine;

  b *= sqrt(b);
  b *= 2.0 + anisotropy2;

  return 3.0 / (8.0 * PI) * a / b;
}

// Compute the density of the atmosphere for a given cartesian position returns the rayleigh density
// as x component and the mie density as y component. The density is assumed to decay exponentially.
vec2 _getDensity(vec3 position) {
  float height = max(0.0, length(position) - PLANET_RADIUS);
  return exp(vec2(-height) / vec2(HEIGHT_R, HEIGHT_M));
}

// Returns the optical depth (e.g. integrated density) between two cartesian points. The ray is
// defined by its origin and direction. The two points are defined by two T parameters along the
// ray. Two values are returned, the rayleigh depth and the mie depth.
vec2 _getOpticalDepth(vec3 camera, vec3 viewRay, float tStart, float tEnd) {
  float fStep = (tEnd - tStart) / SECONDARY_RAY_STEPS;
  vec2  sum   = vec2(0.0);

  for (int i = 0; i < SECONDARY_RAY_STEPS; i++) {
    float tCurr    = tStart + (i + 0.5) * fStep;
    vec3  position = camera + viewRay * tCurr;
    sum += _getDensity(position);
  }

  return sum * fStep;
}

// Calculates the RGB transmittance based on an optical depth.
vec3 _getTransmittance(vec2 opticalDepth) {
  return exp(-BETA_R * opticalDepth.x - BETA_M * opticalDepth.y);
}

// Calculates the RGB transmittance between the two points along the ray defined by the parameters.
vec3 _getTransmittance(vec3 camera, vec3 viewRay, float tStart, float tEnd) {
  return _getTransmittance(_getOpticalDepth(camera, viewRay, tStart, tEnd));
}

// Compute intersections of a ray with a sphere. Two T parameters are returned -- if no intersection
// is found, the first will larger than the second. The T parameters can be nagative. In this case,
// the intersections are behind the origin (in negative ray direction).
vec2 _intersectSphere(vec3 rayOrigin, vec3 rayDir, float radius) {
  float b   = dot(rayOrigin, rayDir);
  float c   = dot(rayOrigin, rayOrigin) - radius * radius;
  float det = b * b - c;

  if (det < 0.0) {
    return vec2(1, -1);
  }

  det = sqrt(det);
  return vec2(-b - det, -b + det);
}

// Computes the intersections of a ray with the atmosphere.
vec2 _intersectAtmosphere(vec3 rayOrigin, vec3 rayDir) {
  return _intersectSphere(rayOrigin, rayDir, ATMO_RADIUS);
}

// Computes the intersections of a ray with the planet.
vec2 _intersectPlanetsphere(vec3 rayOrigin, vec3 rayDir) {
  return _intersectSphere(rayOrigin, rayDir, PLANET_RADIUS);
}

// Returns the color of the incoming light for any direction and position. The ray is defined by its
// origin and direction. The two points are defined by two T parameters along the ray.
vec3 _getInscatter(
    vec3 camera, vec3 viewRay, float tStart, float tEnd, bool hitsSurface, vec3 sunDirection) {

  // we do not always distribute samples evenly:
  //  - if we do hit the planet's surface, we sample evenly
  //  - if the planet surface is not hit, the sampling density depends on start height, if we are
  //    close to the surface, we will sample more at the beginning of the ray where there is more
  //    dense atmosphere
  float startHeight =
      clamp((length(camera + viewRay * tStart) - PLANET_RADIUS) / (ATMO_RADIUS - PLANET_RADIUS),
          0.0, 1.0);
  const float fMaxExponent = 3.0;
  float       fExponent    = 1.0;

  if (!hitsSurface) {
    fExponent = (1.0 - startHeight) * (fMaxExponent - 1.0) + 1.0;
  }

  float dist = (tEnd - tStart);
  vec3  sumR = vec3(0.0);
  vec3  sumM = vec3(0.0);

  for (float i = 0; i < PRIMARY_RAY_STEPS; i++) {
    float tSegmentBegin = tStart + pow((i + 0.0) / (PRIMARY_RAY_STEPS), fExponent) * dist;
    float tMid          = tStart + pow((i + 0.5) / (PRIMARY_RAY_STEPS), fExponent) * dist;
    float tSegmentEnd   = tStart + pow((i + 1.0) / (PRIMARY_RAY_STEPS), fExponent) * dist;

    vec3  position = camera + viewRay * tMid;
    float tSunExit = _intersectAtmosphere(position, sunDirection).y;

    // Compute the transmittance along the path sun -> sample point -> observer
    vec2 opticalDepth    = _getOpticalDepth(camera, viewRay, tStart, tMid);
    vec2 opticalDepthSun = _getOpticalDepth(position, sunDirection, 0, tSunExit);
    vec3 transmittance   = _getTransmittance(opticalDepthSun + opticalDepth);

    // Accumulate all light contributions.
    vec2 density = _getDensity(position);
    sumR += transmittance * density.x * (tSegmentEnd - tSegmentBegin);
    sumM += transmittance * density.y * (tSegmentEnd - tSegmentBegin);
  }

  // The phase can be evaluated outside the loop becaues the scattering angle is the same for all
  // sample points.
  float cosine    = dot(viewRay, sunDirection);
  vec3  inScatter = sumR * BETA_R * _getPhase(cosine, ANISOTROPY_R) +
                   sumM * BETA_M * _getPhase(cosine, ANISOTROPY_M);

  return inScatter;
}

// -------------------------------------------------------------------------------------- public API

// This model does not support refraction.
bool RefractionSupported() {
  return false;
}

vec3 GetRefractedRay(vec3 camera, vec3 ray, float jitter, out bool hitsGround) {
  hitsGround = false;
  return ray;
}

// Returns the sky luminance (in cd/m^2) along the segment from 'camera' to the nearest
// atmosphere boundary in direction 'viewRay', as well as the transmittance along this segment.
vec3 GetSkyLuminance(vec3 camera, vec3 viewRay, vec3 sunDirection, out vec3 transmittance) {

  // If we do not hit the atmosphere, no light will be attenuated.
  vec2 intersections = _intersectAtmosphere(camera, viewRay);
  if (intersections.x > intersections.y || intersections.y < 0) {
    transmittance = vec3(1.0);
    return vec3(0.0);
  }

  // Crop the ray so that it starts at the observer.
  intersections.x = max(intersections.x, 0.0);

  // Compute the incoming light along the entire ray..
  vec3 inscatter =
      _getInscatter(camera, viewRay, intersections.x, intersections.y, false, sunDirection);

  // Compute the transmittance for the entire ray.
  vec2 opticalDepth = _getOpticalDepth(camera, viewRay, intersections.x, intersections.y);
  transmittance     = _getTransmittance(opticalDepth);

  return inscatter * uSunIlluminance;
}

// Returns the sky luminance (in cd/m^2) along the segment from 'camera' to 'p', as well as the
// transmittance along this segment.
vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 p, vec3 sunDirection, out vec3 transmittance) {

  // In this case, the view ray is defined by the camera position and the surface point.
  vec3  viewRay = p - camera;
  float dist    = length(viewRay);
  viewRay /= dist;

  // If we do not hit the atmosphere, no light will be attenuated.
  vec2 intersections = _intersectAtmosphere(camera, viewRay);
  if (intersections.x > intersections.y || intersections.y < 0) {
    transmittance = vec3(1.0);
    return vec3(0.0);
  }

  // Clamp the T parameters to the origin -> surface line.
  intersections.x = max(intersections.x, 0.0);
  intersections.y = min(intersections.y, dist);

  // Compute the incoming light along the entire ray..
  vec3 inscatter =
      _getInscatter(camera, viewRay, intersections.x, intersections.y, true, sunDirection);

  // Compute the transmittance for the entire ray.
  vec2 opticalDepth = _getOpticalDepth(camera, viewRay, intersections.x, intersections.y);
  transmittance     = _getTransmittance(opticalDepth);

  return inscatter * uSunIlluminance;
}

// Returns the sun and sky illuminance (in lux) received on a surface patch located at 'p'.
vec3 GetSunAndSkyIlluminance(vec3 p, vec3 sunDirection, out vec3 skyIlluminance) {

  // This model cannot effiently compute the sky illuminance.
  skyIlluminance = vec3(0.0);

  float tEnd          = _intersectAtmosphere(p, sunDirection).y;
  vec2  opticalDepth  = _getOpticalDepth(p, sunDirection, 0.0, tEnd);
  vec3  transmittance = _getTransmittance(opticalDepth);

  return transmittance * uSunIlluminance;
}
