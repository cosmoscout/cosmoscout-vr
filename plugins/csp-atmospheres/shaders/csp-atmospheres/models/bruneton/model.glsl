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

uniform sampler2D uPhaseTexture;
uniform sampler2D uTransmittanceTexture;
uniform sampler3D uMultipleScatteringTexture;
uniform sampler3D uSingleAerosolsScatteringTexture;
uniform sampler2D uIrradianceTexture;

// Helpers -----------------------------------------------------------------------------------------

// We start with some helper methods which have not been changed functionality-wise.
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html

const float PI = 3.14159265358979323846;

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

// Transmittance Computation -----------------------------------------------------------------------

// The code below is used to comute the optical depth (or transmittance) from any point in the
// atmosphere towards the top atmosphere boundary.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#transmittance

// The only functional difference is that the density of the air molecules, aerosols, and ozone
// molecules is now sampled from a texture (in getDensity()) instead of analytically computed.

float distanceToTopAtmosphereBoundary(float r, float mu) {
  float discriminant = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS;
  return clampDistance(-r * mu + safeSqrt(discriminant));
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

vec3 phaseFunction(float texture_v, float nu) {
  float theta = acos(nu) / PI; // 0<->1
  return texture2D(uPhaseTexture, vec2(theta, texture_v)).rgb;
}

// Single-Scattering Texture Precomputation --------------------------------------------------------

// The code below is used to store the single scattering (without the phase function applied) in a
// 4D lookup table.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#single_scattering_precomputation

// There is no functional difference to the original code.

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

// Irradiance-Texture Lookup -----------------------------------------------------------------------

// The code below is used to retrieve the Sun and sky irradiance values from any altitude from the
// lookup tables.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#irradiance_lookup

// There is no functional difference to the original code.

vec3 getIrradiance(sampler2D irradianceTexture, float r, float muS) {
  vec2 uv = getIrradianceTextureUvFromRMuS(r, muS);
  return vec3(texture(irradianceTexture, uv));
}

// Combining Single- and Multiple-Scattering Contributions -----------------------------------------

// The code below is used to retrieve single molecule-scattrering + multiple-scattering, and single
// aerosol-scattering contributions from the 4D textures.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#rendering

// The only difference is that we removed the optimized case where (monochrome) aerosol scattering
// is stored in the alpha channel of the scattering texture.

void getCombinedScattering(sampler3D multipleScatteringTexture,
    sampler3D singleAerosolsScatteringTexture, float r, float mu, float muS, float nu,
    bool rayRMuIntersectsGround, out vec3 multipleScattering, out vec3 singleAerosolsScattering) {
  vec4  uvwz      = getScatteringTextureUvwzFromRMuMuSNu(r, mu, muS, nu, rayRMuIntersectsGround);
  float texCoordX = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
  float texX      = floor(texCoordX);
  float lerp      = texCoordX - texX;
  vec3  uvw0      = vec3((texX + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  vec3  uvw1      = vec3((texX + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);

  multipleScattering       = vec3(texture(multipleScatteringTexture, uvw0) * (1.0 - lerp) +
                                  texture(multipleScatteringTexture, uvw1) * lerp);
  singleAerosolsScattering = vec3(texture(singleAerosolsScatteringTexture, uvw0) * (1.0 - lerp) +
                                  texture(singleAerosolsScatteringTexture, uvw1) * lerp);
}

// Commpute Sky Radiance ---------------------------------------------------------------------------

// The code below is used to retrieve the color of the sky from the precomputed textures.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#rendering_sky

// The only difference is that we removed the code for light shafts, as this is currently not
// supported by CosmoScout VR.

vec3 getSkyRadiance(sampler2D transmittanceTexture, sampler3D multipleScatteringTexture,
    sampler3D singleAerosolsScatteringTexture, vec3 camera, vec3 viewRay, vec3 sunDirection,
    out vec3 transmittance) {
  // Compute the distance to the top atmosphere boundary along the view ray, assuming the viewer is
  // in space (or NaN if the view ray does not intersect the atmosphere).
  float r                               = length(camera);
  float rmu                             = dot(camera, viewRay);
  float distanceToTopAtmosphereBoundary = -rmu - sqrt(rmu * rmu - r * r + TOP_RADIUS * TOP_RADIUS);
  // If the viewer is in space and the view ray intersects the atmosphere, move the viewer to the
  // top atmosphere boundary (along the view ray):
  if (distanceToTopAtmosphereBoundary > 0.0) {
    camera = camera + viewRay * distanceToTopAtmosphereBoundary;
    r      = TOP_RADIUS;
    rmu += distanceToTopAtmosphereBoundary;
  } else if (r > TOP_RADIUS) {
    // If the view ray does not intersect the atmosphere, simply return 0.
    transmittance = vec3(1.0);
    return vec3(0.0);
  }
  // Compute the r, mu, muS and nu parameters needed for the texture lookups.
  float mu                     = rmu / r;
  float muS                    = dot(camera, sunDirection) / r;
  float nu                     = dot(viewRay, sunDirection);
  bool  rayRMuIntersectsGround = rayIntersectsGround(r, mu);

  transmittance = rayRMuIntersectsGround
                      ? vec3(0.0)
                      : getTransmittanceToTopAtmosphereBoundary(transmittanceTexture, r, mu);

  vec3 multipleScattering;
  vec3 singleAerosolsScattering;
  getCombinedScattering(multipleScatteringTexture, singleAerosolsScatteringTexture, r, mu, muS, nu,
      rayRMuIntersectsGround, multipleScattering, singleAerosolsScattering);

  return multipleScattering * phaseFunction(MOLECULES_PHASE_FUNCTION_V, nu) +
         singleAerosolsScattering * phaseFunction(AEROSOLS_PHASE_FUNCTION_V, nu);
}

// Arial Perspective -------------------------------------------------------------------------------

// The code below is used to retrieve the amount of inscattered light between two points in the
// atmosphere.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#rendering_aerial_perspective

// We removed the code for light shafts, as this is currently not supported by CosmoScout VR. We
// also disabled the "Hack to avoid rendering artifacts when the sun is below the horizon". With
// this hack, the shadow transition in the dusty atemosphere of Mars becomes very harsh.

vec3 getSkyRadianceToPoint(sampler2D transmittanceTexture, sampler3D multipleScatteringTexture,
    sampler3D singleAerosolsScatteringTexture, vec3 camera, vec3 point, vec3 sunDirection,
    out vec3 transmittance) {
  // Compute the distance to the top atmosphere boundary along the view ray, assuming the viewer is
  // in space (or NaN if the view ray does not intersect the atmosphere).
  vec3  viewRay                         = normalize(point - camera);
  float r                               = length(camera);
  float rmu                             = dot(camera, viewRay);
  float distanceToTopAtmosphereBoundary = -rmu - sqrt(rmu * rmu - r * r + TOP_RADIUS * TOP_RADIUS);
  // If the viewer is in space and the view ray intersects the atmosphere, move the viewer to the
  // top atmosphere boundary (along the view ray):
  if (distanceToTopAtmosphereBoundary > 0.0) {
    camera = camera + viewRay * distanceToTopAtmosphereBoundary;
    r      = TOP_RADIUS;
    rmu += distanceToTopAtmosphereBoundary;
  }

  // Compute the r, mu, muS and nu parameters for the first texture lookup.
  float mu                     = rmu / r;
  float muS                    = dot(camera, sunDirection) / r;
  float nu                     = dot(viewRay, sunDirection);
  float d                      = length(point - camera);
  bool  rayRMuIntersectsGround = rayIntersectsGround(r, mu);

  transmittance = getTransmittance(transmittanceTexture, r, mu, d, rayRMuIntersectsGround);

  vec3 multipleScattering;
  vec3 singleAerosolsScattering;

  getCombinedScattering(multipleScatteringTexture, singleAerosolsScatteringTexture, r, mu, muS, nu,
      rayRMuIntersectsGround, multipleScattering, singleAerosolsScattering);

  // Compute the r, mu, muS and nu parameters for the second texture lookup.
  float rP   = clampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  float muP  = (r * mu + d) / rP;
  float muSP = (r * muS + d * nu) / rP;

  vec3 multipleScatteringP;
  vec3 singleAerosolsScatteringP;

  getCombinedScattering(multipleScatteringTexture, singleAerosolsScatteringTexture, rP, muP, muSP,
      nu, rayRMuIntersectsGround, multipleScatteringP, singleAerosolsScatteringP);

  // Combine the lookup results to get the scattering between camera and point.
  multipleScattering       = multipleScattering - transmittance * multipleScatteringP;
  singleAerosolsScattering = singleAerosolsScattering - transmittance * singleAerosolsScatteringP;

  // Hack to avoid rendering artifacts when the sun is below the horizon.
  // singleAerosolsScattering = singleAerosolsScattering * smoothstep(0.0, 0.01, muS);

  return multipleScattering * phaseFunction(MOLECULES_PHASE_FUNCTION_V, nu) +
         singleAerosolsScattering * phaseFunction(AEROSOLS_PHASE_FUNCTION_V, nu);
}

// Ground ------------------------------------------------------------------------------------------

// The code below is used to retrieve the amount of sunlight and skylight reaching a point.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#rendering_ground

// The only difference with respect to the original implementation is the removal of the "normal"
// parameter. In the original implementation, the method used to premultiply the irradiance with the
// dot product between light direction and surface normal. As this factor is already included in the
// BRDFs used in CosmoCout VR, we have removed this. The result is not identical, but as the
// atmosphere is implemented as a post-processing effect in CosmoScout VR, we currently cannot add
// light to areas which did not receive any sunlight in the first place (the color buffer is black
// in these areas).

vec3 getSunAndSkyIrradiance(sampler2D transmittanceTexture, sampler2D irradianceTexture, vec3 point,
    vec3 sunDirection, out vec3 skyIrradiance) {
  float r   = length(point);
  float muS = dot(point, sunDirection) / r;

  // Indirect irradiance.
  skyIrradiance = getIrradiance(irradianceTexture, r, muS);

  // Direct irradiance.
  return SOLAR_IRRADIANCE * getTransmittanceToSun(transmittanceTexture, r, muS);
}

vec3 GetSkyLuminance(vec3 camera, vec3 viewRay, vec3 sunDirection, out vec3 transmittance) {
  return getSkyRadiance(uTransmittanceTexture, uMultipleScatteringTexture,
             uSingleAerosolsScatteringTexture, camera, viewRay, sunDirection, transmittance) *
         SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
}

vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 point, vec3 sunDirection, out vec3 transmittance) {
  return getSkyRadianceToPoint(uTransmittanceTexture, uMultipleScatteringTexture,
             uSingleAerosolsScatteringTexture, camera, point, sunDirection, transmittance) *
         SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
}

vec3 GetSunAndSkyIlluminance(vec3 p, vec3 sunDirection, out vec3 skyIrradiance) {
  vec3 sun_irradiance = getSunAndSkyIrradiance(
      uTransmittanceTexture, uIrradianceTexture, p, sunDirection, skyIrradiance);
  skyIrradiance *= SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
  return sun_irradiance * SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
}