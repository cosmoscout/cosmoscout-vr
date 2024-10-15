////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-FileCopyrightText: 2008 INRIA
// SPDX-License-Identifier: BSD-3-Clause

// This file is based on the original implementation by Eric Bruneton:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl

// While implementing the atmospheric model into CosmoScout VR, we have refactored some parts of the
// code, however this is mostly related to how variables are named and how input parameters are
// passed to the shader. The only fundamental change is that the phase functions for aerosols and
// molecules as well as their density distributions are now sampled from textures.

// Below, we will indicate for each group of function whether something has been changed and a link
// to the original explanations of the methods by Eric Bruneton.

// Helpers -----------------------------------------------------------------------------------------

// We start with some helper methods which have not been changed functionality-wise.
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html

float clampCosine(float mu) {
  return clamp(mu, -1.0, 1.0);
}

float clampDistance(float d) {
  return max(d, 0.0);
}

float clampRadius(float r) {
  return clamp(r, BOTTOM_RADIUS, TOP_RADIUS);
}

float safeSqrt(float a) {
  return sqrt(max(a, 0.0));
}

float distanceToTopAtmosphereBoundary(float r, float mu) {
  float discriminant = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS;
  return clampDistance(-r * mu + safeSqrt(discriminant));
}

float distanceToBottomAtmosphereBoundary(float r, float mu) {
  float discriminant = r * r * (mu * mu - 1.0) + BOTTOM_RADIUS * BOTTOM_RADIUS;
  return clampDistance(-r * mu - safeSqrt(discriminant));
}

bool rayIntersectsGround(float r, float mu) {
  return mu < 0.0 && r * r * (mu * mu - 1.0) + BOTTOM_RADIUS * BOTTOM_RADIUS >= 0.0;
}

// Transmittance Texture Precomputation ------------------------------------------------------------

// The code below is used to store the precomputed transmittance values in a 2D lookup table.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#transmittance_precomputation

// There is no functional difference to the original code.

float getTextureCoordFromUnitRange(float x, int textureSize) {
  return 0.5 / float(textureSize) + x * (1.0 - 1.0 / float(textureSize));
}

float getUnitRangeFromTextureCoord(float u, int textureSize) {
  return (u - 0.5 / float(textureSize)) / (1.0 - 1.0 / float(textureSize));
}

vec2 getTransmittanceTextureUvFromRMu(float r, float mu) {
  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon.
  float rho = safeSqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
  // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
  float d    = distanceToTopAtmosphereBoundary(r, mu);
  float dMin = TOP_RADIUS - r;
  float dMax = rho + H;
  float xMu  = (d - dMin) / (dMax - dMin);
  float xR   = rho / H;
  return vec2(getTextureCoordFromUnitRange(xMu, TRANSMITTANCE_TEXTURE_WIDTH),
      getTextureCoordFromUnitRange(xR, TRANSMITTANCE_TEXTURE_HEIGHT));
}

void getRMuFromTransmittanceTextureUv(vec2 uv, out float r, out float mu) {
  float xMu = getUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
  float xR  = getUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon, from which we can compute r:
  float rho = H * xR;
  r         = sqrt(rho * rho + BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
  // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon) -
  // from which we can recover mu:
  float dMin = TOP_RADIUS - r;
  float dMax = rho + H;
  float d    = dMin + xMu * (dMax - dMin);
  mu         = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
  mu         = clampCosine(mu);
}

// Transmittance Texture Lookup --------------------------------------------------------------------

// The code below is used to retrieve the transmittance values from the lookup tables.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#transmittance_lookup

// There is no functional difference to the original code.

vec3 getTransmittanceToTopAtmosphereBoundary(sampler2D transmittanceTexture, float r, float mu) {
  vec2 uv = getTransmittanceTextureUvFromRMu(r, mu);
  return vec3(texture(transmittanceTexture, uv));
}

vec3 getTransmittance(
    sampler2D transmittanceTexture, float r, float mu, float d, bool rayRMuIntersectsGround) {

  float rD  = clampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  float muD = clampCosine((r * mu + d) / rD);

  if (rayRMuIntersectsGround) {
    return min(getTransmittanceToTopAtmosphereBoundary(transmittanceTexture, rD, -muD) /
                   getTransmittanceToTopAtmosphereBoundary(transmittanceTexture, r, -mu),
        vec3(1.0));
  } else {
    return min(getTransmittanceToTopAtmosphereBoundary(transmittanceTexture, r, mu) /
                   getTransmittanceToTopAtmosphereBoundary(transmittanceTexture, rD, muD),
        vec3(1.0));
  }
}

vec3 getTransmittanceToSun(sampler2D transmittanceTexture, float r, float muS) {
  float sinThetaH = BOTTOM_RADIUS / r;
  float cosThetaH = -sqrt(max(1.0 - sinThetaH * sinThetaH, 0.0));
  return getTransmittanceToTopAtmosphereBoundary(transmittanceTexture, r, muS) *
         smoothstep(
             -sinThetaH * SUN_ANGULAR_RADIUS, sinThetaH * SUN_ANGULAR_RADIUS, muS - cosThetaH);
}

vec4 getScatteringTextureUvwzFromRMuMuSNu(
    float r, float mu, float muS, float nu, bool rayRMuIntersectsGround) {

  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon.
  float rho = safeSqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS);
  float u_r = getTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

  // Discriminant of the quadratic equation for the intersections of the ray (r,mu) with the ground
  // (see rayIntersectsGround).
  float rMu          = r * mu;
  float discriminant = rMu * rMu - r * r + BOTTOM_RADIUS * BOTTOM_RADIUS;
  float uMu;
  if (rayRMuIntersectsGround) {
    // Distance to the ground for the ray (r,mu), and its minimum and maximum values over all mu -
    // obtained for (r,-1) and (r,mu_horizon).
    float d    = -rMu - safeSqrt(discriminant);
    float dMin = r - BOTTOM_RADIUS;
    float dMax = rho;
    uMu = 0.5 - 0.5 * getTextureCoordFromUnitRange(dMax == dMin ? 0.0 : (d - dMin) / (dMax - dMin),
                          SCATTERING_TEXTURE_MU_SIZE / 2);
  } else {
    // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum and maximum
    // values over all mu - obtained for (r,1) and (r,mu_horizon).
    float d    = -rMu + safeSqrt(discriminant + H * H);
    float dMin = TOP_RADIUS - r;
    float dMax = rho + H;
    uMu        = 0.5 + 0.5 * getTextureCoordFromUnitRange(
                          (d - dMin) / (dMax - dMin), SCATTERING_TEXTURE_MU_SIZE / 2);
  }

  float d    = distanceToTopAtmosphereBoundary(BOTTOM_RADIUS, muS);
  float dMin = TOP_RADIUS - BOTTOM_RADIUS;
  float dMax = H;
  float a    = (d - dMin) / (dMax - dMin);
  float D    = distanceToTopAtmosphereBoundary(BOTTOM_RADIUS, MU_S_MIN);
  float A    = (D - dMin) / (dMax - dMin);
  // An ad-hoc function equal to 0 for muS = MU_S_MIN (because then d = D and thus a = A), equal to
  // 1 for muS = 1 (because then d = dMin and thus a = 0), and with a large slope around muS = 0,
  // to get more texture samples near the horizon.
  float uMuS =
      getTextureCoordFromUnitRange(max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

  float uNu = (nu + 1.0) / 2.0;
  return vec4(uNu, uMuS, uMu, u_r);
}

void getRMuMuSNuFromScatteringTextureUvwz(vec4 uvwz, out float r, out float mu, out float muS,
    out float nu, out bool rayRMuIntersectsGround) {

  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon.
  float rho = H * getUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
  r         = sqrt(rho * rho + BOTTOM_RADIUS * BOTTOM_RADIUS);

  if (uvwz.z < 0.5) {
    // Distance to the ground for the ray (r,mu), and its minimum and maximum values over all mu -
    // obtained for (r,-1) and (r,mu_horizon) - from which we can recover mu:
    float dMin = r - BOTTOM_RADIUS;
    float dMax = rho;
    float d    = dMin + (dMax - dMin) * getUnitRangeFromTextureCoord(
                                         1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);
    mu                     = d == 0.0 ? -1.0 : clampCosine(-(rho * rho + d * d) / (2.0 * r * d));
    rayRMuIntersectsGround = true;
  } else {
    // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum and maximum
    // values over all mu - obtained for (r,1) and (r,mu_horizon) - from which we can recover mu:
    float dMin = TOP_RADIUS - r;
    float dMax = rho + H;
    float d    = dMin + (dMax - dMin) * getUnitRangeFromTextureCoord(
                                         2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);
    mu = d == 0.0 ? 1.0 : clampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
    rayRMuIntersectsGround = false;
  }

  float xMuS = getUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
  float dMin = TOP_RADIUS - BOTTOM_RADIUS;
  float dMax = H;
  float D    = distanceToTopAtmosphereBoundary(BOTTOM_RADIUS, MU_S_MIN);
  float A    = (D - dMin) / (dMax - dMin);
  float a    = (A - xMuS * A) / (1.0 + xMuS * A);
  float d    = dMin + min(a, A) * (dMax - dMin);
  muS        = d == 0.0 ? 1.0 : clampCosine((H * H - d * d) / (2.0 * BOTTOM_RADIUS * d));

  nu = clampCosine(uvwz.x * 2.0 - 1.0);
}

// Irradiance-Texture Precomputation ---------------------------------------------------------------

// The code below is used to store the direct and indirect irradiance received at any altitude in 2D
// lookup tables.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#irradiance_precomputation

// There is no functional difference to the original code.

vec2 getIrradianceTextureUvFromRMuS(float r, float muS) {
  float xR   = (r - BOTTOM_RADIUS) / (TOP_RADIUS - BOTTOM_RADIUS);
  float xMuS = muS * 0.5 + 0.5;
  return vec2(getTextureCoordFromUnitRange(xMuS, IRRADIANCE_TEXTURE_WIDTH),
      getTextureCoordFromUnitRange(xR, IRRADIANCE_TEXTURE_HEIGHT));
}

void getRMuSFromIrradianceTextureUv(vec2 uv, out float r, out float muS) {
  float xMuS = getUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
  float xR   = getUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
  r          = BOTTOM_RADIUS + xR * (TOP_RADIUS - BOTTOM_RADIUS);
  muS        = clampCosine(2.0 * xMuS - 1.0);
}
