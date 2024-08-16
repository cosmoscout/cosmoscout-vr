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

const float PI = 3.14159265358979323846;

uniform sampler2D uPhaseTexture;
uniform sampler2D uTransmittanceTexture;
uniform sampler3D uMultipleScatteringTexture;
uniform sampler3D uSingleAerosolsScatteringTexture;
uniform sampler2D uIrradianceTexture;

#if USE_REFRACTION
uniform sampler2D uThetaDeviationTexture;
#endif

vec3 moleculePhaseFunction(float nu) {
  float theta = acos(nu) / PI; // 0<->1
  return texture2D(uPhaseTexture, vec2(theta, 0.0)).rgb;
}

vec3 aerosolPhaseFunction(float nu) {
  float theta = acos(nu) / PI; // 0<->1
  return texture2D(uPhaseTexture, vec2(theta, 1.0)).rgb;
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

  return multipleScattering * moleculePhaseFunction(nu) +
         singleAerosolsScattering * aerosolPhaseFunction(nu);
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

  // Avoid negative values due to precision errors.
  multipleScattering       = max(multipleScattering, vec3(0.0));
  singleAerosolsScattering = max(singleAerosolsScattering, vec3(0.0));

  // Hack to avoid rendering artifacts when the sun is below the horizon.
  // singleAerosolsScattering = singleAerosolsScattering * smoothstep(0.0, 0.01, muS);

  return multipleScattering * moleculePhaseFunction(nu) +
         singleAerosolsScattering * aerosolPhaseFunction(nu);
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
  vec2 uv       = getIrradianceTextureUvFromRMuS(r, muS);
  skyIrradiance = vec3(texture(irradianceTexture, uv));

  // Direct irradiance.
  return SOLAR_ILLUMINANCE * getTransmittanceToSun(transmittanceTexture, r, muS);
}

// Rodrigues' rotation formula
vec3 rotateVector2(vec3 v, vec3 a, float sinMu) {
  float cosMu = sqrt(1.0 - sinMu * sinMu);
  return v * cosMu + cross(a, v) * sinMu + a * dot(a, v) * (1.0 - cosMu);
}

// Public API --------------------------------------------------------------------------------------

bool RefractionSupported() {
#if USE_REFRACTION
  return true;
#else
  return false;
#endif
}

vec3 GetRefractedRay(vec3 camera, vec3 ray, out bool hitsGround) {
#if USE_REFRACTION
  float r  = length(camera);
  float mu = dot(camera / r, ray);
  vec2  uv = getTransmittanceTextureUvFromRMu(r, mu);

  vec2  deviationContactRadius = texture(uThetaDeviationTexture, uv).rg;
  float sinMu                  = sin(deviationContactRadius.r);
  vec3  axis                   = normalize(cross(camera, ray));

  hitsGround = deviationContactRadius.g < 0.0;

  return rotateVector2(ray, axis, sinMu);

#else
  hitsGround = false;
  return ray;
#endif
}

vec3 GetSkyLuminance(vec3 camera, vec3 viewRay, vec3 sunDirection, out vec3 transmittance) {
  return getSkyRadiance(uTransmittanceTexture, uMultipleScatteringTexture,
      uSingleAerosolsScatteringTexture, camera, viewRay, sunDirection, transmittance);
}

vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 point, vec3 sunDirection, out vec3 transmittance) {
  return getSkyRadianceToPoint(uTransmittanceTexture, uMultipleScatteringTexture,
      uSingleAerosolsScatteringTexture, camera, point, sunDirection, transmittance);
}

vec3 GetSunAndSkyIlluminance(vec3 p, vec3 sunDirection, out vec3 skyIrradiance) {
  vec3 sun_irradiance = getSunAndSkyIrradiance(
      uTransmittanceTexture, uIrradianceTexture, p, sunDirection, skyIrradiance);
  return sun_irradiance;
}