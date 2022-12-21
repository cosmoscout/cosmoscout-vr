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
float _getPhase(float fCosine, float fAnisotropy) {
  float fAnisotropy2 = fAnisotropy * fAnisotropy;
  float fCosine2     = fCosine * fCosine;

  float a = (1.0 - fAnisotropy2) * (1.0 + fCosine2);
  float b = 1.0 + fAnisotropy2 - 2.0 * fAnisotropy * fCosine;

  b *= sqrt(b);
  b *= 2.0 + fAnisotropy2;

  return 3.0 / (8.0 * PI) * a / b;
}

// compute the density of the atmosphere for a given model space position
// returns the rayleigh density as x component and the mie density as Y
vec2 _getDensity(vec3 vPos) {
  float fHeight = max(0.0, length(vPos) - PLANET_RADIUS);
  return exp(vec2(-fHeight) / vec2(HEIGHT_R, HEIGHT_M));
}

// returns the optical depth between two points in model space
// The ray is defined by its origin and direction. The two points are defined
// by two T parameters along the ray. Two values are returned, the rayleigh
// depth and the mie depth.
vec2 _getOpticalDepth(vec3 camera, vec3 viewRay, float fTStart, float fTEnd) {
  float fStep = (fTEnd - fTStart) / SECONDARY_RAY_STEPS;
  vec2  vSum  = vec2(0.0);

  for (int i = 0; i < SECONDARY_RAY_STEPS; i++) {
    float fTCurr = fTStart + (i + 0.5) * fStep;
    vec3  vPos   = camera + viewRay * fTCurr;
    vSum += _getDensity(vPos);
  }

  return vSum * fStep;
}

// calculates the extinction based on an optical depth
vec3 _getExtinction(vec2 vOpticalDepth) {
  return exp(-BETA_R * vOpticalDepth.x - BETA_M * vOpticalDepth.y);
}

// returns the irradiance for the current pixel
// This is based on the color buffer and the extinction of light.
vec3 _getExtinction(vec3 camera, vec3 viewRay, float fTStart, float fTEnd) {
  vec2 vOpticalDepth = _getOpticalDepth(camera, viewRay, fTStart, fTEnd);
  return _getExtinction(vOpticalDepth);
}

// compute intersections with the atmosphere
// two T parameters are returned -- if no intersection is found, the first will
// larger than the second
vec2 _intersectSphere(vec3 camera, vec3 viewRay, float fRadius) {
  float b    = dot(camera, viewRay);
  float c    = dot(camera, camera) - fRadius * fRadius;
  float fDet = b * b - c;

  if (fDet < 0.0) {
    return vec2(1, -1);
  }

  fDet = sqrt(fDet);
  return vec2(-b - fDet, -b + fDet);
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
    vec3 camera, vec3 viewRay, float fTStart, float fTEnd, bool bHitsSurface, vec3 sunDirection) {
  // we do not always distribute samples evenly:
  //  - if we do hit the planet's surface, we sample evenly
  //  - if the planet surface is not hit, the sampling density depends on start height, if we are
  //    close to the surface, we will sample more at the beginning of the ray where there is more
  //    dense atmosphere
  float fstartHeight =
      clamp((length(camera + viewRay * fTStart) - PLANET_RADIUS) / (ATMO_RADIUS - PLANET_RADIUS),
          0.0, 1.0);
  const float fMaxExponent = 3.0;
  float       fExponent    = 1.0;

  if (!bHitsSurface) {
    fExponent = (1.0 - fstartHeight) * (fMaxExponent - 1.0) + 1.0;
  }

  float fDist = (fTEnd - fTStart);
  vec3  sumR  = vec3(0.0);
  vec3  sumM  = vec3(0.0);

  for (float i = 0; i < PRIMARY_RAY_STEPS; i++) {
    float fTSegmentBegin = fTStart + pow((i + 0.0) / (PRIMARY_RAY_STEPS), fExponent) * fDist;
    float fTMid          = fTStart + pow((i + 0.5) / (PRIMARY_RAY_STEPS), fExponent) * fDist;
    float fTSegmentEnd   = fTStart + pow((i + 1.0) / (PRIMARY_RAY_STEPS), fExponent) * fDist;

    vec3  vPos      = camera + viewRay * fTMid;
    float fTSunExit = _intersectAtmosphere(vPos, sunDirection).y;

    vec2 vOpticalDepth    = _getOpticalDepth(camera, viewRay, fTStart, fTMid);
    vec2 vOpticalDepthSun = _getOpticalDepth(vPos, sunDirection, 0, fTSunExit);
    vec3 vExtinction      = _getExtinction(vOpticalDepthSun + vOpticalDepth);

    vec2 vDensity = _getDensity(vPos);

    sumR += vExtinction * vDensity.x * (fTSegmentEnd - fTSegmentBegin);
    sumM += vExtinction * vDensity.y * (fTSegmentEnd - fTSegmentBegin);
  }

  float fCosine    = dot(viewRay, sunDirection);
  vec3  vInScatter = sumR * BETA_R * _getPhase(fCosine, ANISOTROPY_R) +
                    sumM * BETA_M * _getPhase(fCosine, ANISOTROPY_M);

  return vInScatter;
}

// -------------------------------------------------------------------------------------- public API

// Returns the sky luminance along the segment from 'camera' to the nearest atmosphere boundary in
// direction 'viewRay', as well as the transmittance along this segment.
vec3 GetSkyLuminance(
    vec3 camera, vec3 viewRay, float shadowLength, vec3 sunDirection, out vec3 transmittance) {

  float fTEnd     = _intersectAtmosphere(camera, viewRay).y;
  vec3  inscatter = _getInscatter(camera, viewRay, 0.0, fTEnd, false, sunDirection);

  vec2 vOpticalDepth = _getOpticalDepth(camera, viewRay, 0.0, fTEnd);
  transmittance      = _getExtinction(vOpticalDepth);

  return inscatter * uSunIlluminance;
}

// Returns the sky luminance along the segment from 'camera' to 'p', as well as the transmittance
// along this segment.
vec3 GetSkyLuminanceToPoint(
    vec3 camera, vec3 p, float shadowLength, vec3 sunDirection, out vec3 transmittance) {

  vec3  viewRay = p - camera;
  float fTEnd   = length(viewRay);

  vec3 inscatter = _getInscatter(camera, viewRay / fTEnd, 0.0, fTEnd, true, sunDirection);

  vec2 vOpticalDepth = _getOpticalDepth(camera, viewRay / fTEnd, 0.0, fTEnd);
  transmittance      = _getExtinction(vOpticalDepth);

  return inscatter * uSunIlluminance;
}

// Returns the sun and sky illuminance received on a surface patch located at 'p' and whose normal
// vector is 'normal'.
vec3 GetSunAndSkyIlluminance(vec3 p, vec3 normal, vec3 sunDirection, out vec3 skyIlluminance) {
  skyIlluminance = vec3(0.0);
  return vec3(uSunIlluminance);
}
