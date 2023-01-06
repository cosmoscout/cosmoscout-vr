#version 330

// constants
const float PI = 3.14159265359;

uniform float uSunIlluminance;

// returns the probability of scattering
// based on the cosine (c) between in and out direction and the anisotropy (g)
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

// compute the density of the atmosphere for a given model space position
// returns the rayleigh density as x component and the mie density as Y
vec2 _getDensity(vec3 position) {
  float height = max(0.0, length(position) - PLANET_RADIUS);
  return exp(vec2(-height) / vec2(HEIGHT_R, HEIGHT_M));
}

// returns the optical depth between two points in model space
// The ray is defined by its origin and direction. The two points are defined
// by two T parameters along the ray. Two values are returned, the rayleigh
// depth and the mie depth.
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

// calculates the extinction based on an optical depth
vec3 _getExtinction(vec2 opticalDepth) {
  return exp(-BETA_R * opticalDepth.x - BETA_M * opticalDepth.y);
}

// returns the irradiance for the current pixel
// This is based on the color buffer and the extinction of light.
vec3 _getExtinction(vec3 camera, vec3 viewRay, float tStart, float tEnd) {
  vec2 opticalDepth = _getOpticalDepth(camera, viewRay, tStart, tEnd);
  return _getExtinction(opticalDepth);
}

// compute intersections with the atmosphere
// two T parameters are returned -- if no intersection is found, the first will
// larger than the second
vec2 _intersectSphere(vec3 camera, vec3 viewRay, float radius) {
  float b   = dot(camera, viewRay);
  float c   = dot(camera, camera) - radius * radius;
  float det = b * b - c;

  if (det < 0.0) {
    return vec2(1, -1);
  }

  det = sqrt(det);
  return vec2(-b - det, -b + det);
}

vec2 _intersectAtmosphere(vec3 camera, vec3 viewRay) {
  return _intersectSphere(camera, viewRay, ATMO_RADIUS);
}

vec2 _intersectPlanetsphere(vec3 camera, vec3 viewRay) {
  return _intersectSphere(camera, viewRay, PLANET_RADIUS);
}

// returns the color of the incoming light for any direction and position
// The ray is defined by its origin and direction. The two points are defined
// by two T parameters along the ray. Everything is in model space.
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

    vec2 opticalDepth    = _getOpticalDepth(camera, viewRay, tStart, tMid);
    vec2 opticalDepthSun = _getOpticalDepth(position, sunDirection, 0, tSunExit);
    vec3 extinction      = _getExtinction(opticalDepthSun + opticalDepth);

    vec2 density = _getDensity(position);

    sumR += extinction * density.x * (tSegmentEnd - tSegmentBegin);
    sumM += extinction * density.y * (tSegmentEnd - tSegmentBegin);
  }

  float cosine    = dot(viewRay, sunDirection);
  vec3  inScatter = sumR * BETA_R * _getPhase(cosine, ANISOTROPY_R) +
                   sumM * BETA_M * _getPhase(cosine, ANISOTROPY_M);

  return inScatter;
}

// -------------------------------------------------------------------------------------- public API

// Returns the sky luminance along the segment from 'camera' to the nearest atmosphere boundary in
// direction 'viewRay', as well as the transmittance along this segment.
vec3 GetSkyLuminance(
    vec3 camera, vec3 viewRay, float shadowLength, vec3 sunDirection, out vec3 transmittance) {

  vec2 intersections = _intersectAtmosphere(camera, viewRay);

  if (intersections.x > intersections.y || intersections.y < 0) {
    transmittance = vec3(1.0);
    return vec3(0.0);
  }

  intersections.x = max(intersections.x, 0.0);

  vec3 inscatter =
      _getInscatter(camera, viewRay, intersections.x, intersections.y, false, sunDirection);

  vec2 opticalDepth = _getOpticalDepth(camera, viewRay, intersections.x, intersections.y);
  transmittance     = _getExtinction(opticalDepth);

  return inscatter * uSunIlluminance;
}

// Returns the sky luminance along the segment from 'camera' to 'p', as well as the transmittance
// along this segment.
vec3 GetSkyLuminanceToPoint(
    vec3 camera, vec3 p, float shadowLength, vec3 sunDirection, out vec3 transmittance) {

  vec3  viewRay = p - camera;
  float dist    = length(viewRay);
  viewRay /= dist;

  vec2 intersections = _intersectAtmosphere(camera, viewRay);

  if (intersections.x > intersections.y || intersections.y < 0) {
    transmittance = vec3(1.0);
    return vec3(0.0);
  }

  intersections.x = max(intersections.x, 0.0);
  intersections.y = min(intersections.y, dist);

  vec3 inscatter =
      _getInscatter(camera, viewRay, intersections.x, intersections.y, true, sunDirection);

  vec2 opticalDepth = _getOpticalDepth(camera, viewRay, intersections.x, intersections.y);
  transmittance     = _getExtinction(opticalDepth);

  return inscatter * uSunIlluminance;
}

// Returns the sun and sky illuminance received on a surface patch located at 'p' and whose normal
// vector is 'normal'.
vec3 GetSunAndSkyIlluminance(vec3 p, vec3 normal, vec3 sunDirection, out vec3 skyIlluminance) {
  vec3 transmittance;
  skyIlluminance = GetSkyLuminance(p, normal, 0.0, sunDirection, transmittance);

  float tEnd         = _intersectAtmosphere(p, sunDirection).y;
  vec2  opticalDepth = _getOpticalDepth(p, sunDirection, 0.0, tEnd);
  transmittance      = _getExtinction(opticalDepth);

  return transmittance * uSunIlluminance;
}
