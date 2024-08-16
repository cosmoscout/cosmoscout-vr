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
// passed to the shader. The are only two fundamental changes:
//   1. The phase functions for aerosols and molecules as well as their density distributions are
//      now sampled from textures.
//   2. The index of refraction of the atmosphere is now passed to the shader and rays are refracted
//      accordingly.

// Below, we will indicate for each group of function whether something important has been changed
// and a link to the original explanations of the methods by Eric Bruneton.

// Refraction Computation --------------------------------------------------------------------------

// This part is new. The wavelength-dependent INDEX_OF_REFRACTION is passed to the shader. According
// to the Gladsstone-Dale relation, the refractive index is proportional to the density of the air
// molecules. The density of the air molecules and aerosols is sampled from a texture.

float getDensity(float densityTextureV, float altitude) {
  float u = clamp(altitude / (TOP_RADIUS - BOTTOM_RADIUS), 0.0, 1.0);
  return texture(uDensityTexture, vec2(u, densityTextureV)).r;
}

// Transmittance Computation -----------------------------------------------------------------------

// The code below is used to comute the optical depth (or transmittance) from any point in the
// atmosphere towards the top atmosphere boundary.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#transmittance

// The only functional difference is that the density of the air molecules, aerosols, and ozone
// molecules is now sampled from a texture (in getDensity()) instead of analytically computed.

#if COMPUTE_REFRACTION

float getRefractiveIndexMinusOne(float altitude) {
  return INDEX_OF_REFRACTION * getDensity(ATMOSPHERE.molecules.densityTextureV, altitude);
}

double getRefractiveIndex(float altitude) {
  return 1.0lf + double(getRefractiveIndexMinusOne(altitude));
}

float getRefractiveIndexGradientLength(float height, float dh) {
  return (getRefractiveIndexMinusOne(height + dh) - getRefractiveIndexMinusOne(height)) / dh;
}

#line 61

dvec2 rayGradient(dvec2 rayOrigin, dvec2 rayDir) {
  double altitude        = length(rayOrigin);
  double refractiveIndex = getRefractiveIndex(float(altitude) - BOTTOM_RADIUS);
  double gradientLength  = getRefractiveIndexGradientLength(float(altitude) - BOTTOM_RADIUS, 100);
  return rayOrigin / altitude * gradientLength;
}

dvec2 refractRay(dvec2 rayOrigin, dvec2 rayDir, double stepLength) {
  double altitude        = length(rayOrigin);
  double refractiveIndex = getRefractiveIndex(float(altitude) - BOTTOM_RADIUS);
  double gradientLength  = getRefractiveIndexGradientLength(float(altitude) - BOTTOM_RADIUS, 100);
  dvec2  dn              = rayOrigin / altitude * gradientLength;
  return normalize(refractiveIndex * rayDir + dn * stepLength);
}

dvec2 refractRayRungeKutta(dvec2 rayOrigin, dvec2 rayDir, double stepLength) {
  dvec2 k1 = stepLength * rayGradient(rayOrigin, rayDir);
  dvec2 k2 = stepLength * rayGradient(rayOrigin + 0.5 * stepLength * rayDir, rayDir + 0.5 * k1);
  dvec2 k3 =
      stepLength * rayGradient(rayOrigin + 0.5 * stepLength * rayDir + 0.5 * k2, rayDir + 0.5 * k2);
  dvec2 k4 = stepLength * rayGradient(rayOrigin + stepLength * rayDir, rayDir + k3);

  return normalize(rayDir + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0);
}

// If our ray hit the surface of the planet, the optical depth should be infinite. However,
// returning a very high optical depth is not a good solution, because the transmittance
// computation in getTransmittance() in common.glsl relies on the fact that the ray does not
// intersect the planet in either the forward or backward direction.
// As a solution, we simply compute the optical depth as if the ray would have continued close
// to the surface of the planet. This means, that there will be a very small region above the
// horizon where the Sun should not be visible but will be visible.
// We accept this small error during real-time rendering, however for the eclipse-shadow
// computation, we will require the information about the ground intersection.
struct RayInfo {
  vec3  opticalDepth;
  float thetaDeviation;
  float contactRadius;
};

// Using acos is not very stable for small angles. This function is used to compute the angle
// between two normalized vectors in a more stable way.
float angleBetweenVectors(vec2 u, vec2 v) {
  return 2.0 * asin(0.5 * length(u - v));
}

// If refraction is incorporated, the ray is bent according to the refractive index of the air
// molecules. The optical depth is then computed by integrating the density along the refracted
// ray.
RayInfo computeOpticalLengthToTopAtmosphereBoundary(float densityTextureV, float r, float mu) {

  double dx          = STEP_SIZE_OPTICAL_DEPTH;
  dvec2  startRayDir = vec2(sqrt(1 - mu * mu), mu);

  RayInfo result;
  result.opticalDepth   = vec3(0.0);
  result.thetaDeviation = 0.0;
  result.contactRadius  = r - BOTTOM_RADIUS;

  dvec2  samplePos    = vec2(0.0, r);
  double sampleRadius = r;
  dvec2  currentDir   = startRayDir;

  while (sampleRadius <= TOP_RADIUS) {

    samplePos += currentDir * dx;
    sampleRadius = length(samplePos);

    float altitude = float(sampleRadius) - BOTTOM_RADIUS;
    result.opticalDepth += getDensity(densityTextureV, altitude);
    result.contactRadius = min(result.contactRadius, altitude);

    // If the ray intersects the ground, we have a problem: We do not have density information
    // in the underground, so the ray traversal will become undefined. Hence we clamp the ray
    // to the surface of the planet.
    if (sampleRadius > BOTTOM_RADIUS) {
      currentDir = refractRayRungeKutta(samplePos, currentDir, dx);
    }
  }

  result.thetaDeviation = angleBetweenVectors(vec2(startRayDir), vec2(currentDir));
  result.opticalDepth *= float(dx);

  return result;
}

#else

struct RayInfo {
  float opticalDepth;
};

// If no refraction is computed, the optical depth is simply the integral of the density along the
// ray. We use the original implementation by Eric Bruneton with evenly spaced samples along the
// ray.
RayInfo computeOpticalLengthToTopAtmosphereBoundary(float densityTextureV, float r, float mu) {
  float   dx = distanceToTopAtmosphereBoundary(r, mu) / float(SAMPLE_COUNT_OPTICAL_DEPTH);
  RayInfo result;
  result.opticalDepth = 0.0;

  for (int i = 0; i <= SAMPLE_COUNT_OPTICAL_DEPTH; ++i) {
    float d_i      = float(i) * dx;
    float r_i      = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
    float y_i      = getDensity(densityTextureV, r_i - BOTTOM_RADIUS);
    float weight_i = i == 0 || i == SAMPLE_COUNT_OPTICAL_DEPTH ? 0.5 : 1.0;
    result.opticalDepth += y_i * weight_i * dx;
  }
  return result;
}

#endif

// Transmittance Texture Precomputation ------------------------------------------------------------

// The code below is used to store the precomputed transmittance values in a 2D lookup table.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#transmittance_precomputation

// There is no functional difference to the original code.

#if COMPUTE_REFRACTION

vec3 computeTransmittanceToTopAtmosphereBoundaryTexture(AtmosphereComponents atmosphere,
    vec2 fragCoord, out float thetaDeviation, out float contactRadius) {

#else

vec3 computeTransmittanceToTopAtmosphereBoundaryTexture(
    AtmosphereComponents atmosphere, vec2 fragCoord) {

#endif

  float r;
  float mu;
  getRMuFromTransmittanceTextureUv(
      fragCoord / vec2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT), r, mu);

  RayInfo molecules =
      computeOpticalLengthToTopAtmosphereBoundary(atmosphere.molecules.densityTextureV, r, mu);
  RayInfo aerosols =
      computeOpticalLengthToTopAtmosphereBoundary(atmosphere.aerosols.densityTextureV, r, mu);
  RayInfo ozone =
      computeOpticalLengthToTopAtmosphereBoundary(atmosphere.ozone.densityTextureV, r, mu);

  vec3 transmittance = exp(-(atmosphere.molecules.extinction * molecules.opticalDepth +
                             atmosphere.aerosols.extinction * aerosols.opticalDepth +
                             atmosphere.ozone.extinction * ozone.opticalDepth));

#if COMPUTE_REFRACTION
  thetaDeviation = molecules.thetaDeviation;
  contactRadius  = molecules.contactRadius;
#endif

  return transmittance;
}

// Single-Scattering Computation -------------------------------------------------------------------

// The code below is used to compute the amount of light scattered into a specific direction during
// a single scattering event for air molecules and aerosols.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#single_scattering

// Most of the methods below are functionality-wise identical to the original implementation. The
// only difference is that the RayleighPhaseFunction() and MiePhaseFunction() have been removed and
// replaced by a generic phaseFunction() which samples the phase function from a texture.

float distanceToNearestAtmosphereBoundary(float r, float mu, bool rayRMuIntersectsGround) {
  if (rayRMuIntersectsGround) {
    return distanceToBottomAtmosphereBoundary(r, mu);
  } else {
    return distanceToTopAtmosphereBoundary(r, mu);
  }
}

#if COMPUTE_REFRACTION

// As for the transmittance, the single-scattering computation is different if refraction is
// incorporated. The ray is bent, so we cannot simply accumulate the scattering along a straight
// line.
void computeSingleScattering(AtmosphereComponents atmosphere, sampler2D transmittanceTexture,
    float r, float mu, float muS, float nu, bool rayRMuIntersectsGround, out vec3 molecules,
    out vec3 aerosols) {

  // The integration step, i.e. the length of each integration interval.
  // double dx = double(TOP_RADIUS) / SAMPLE_COUNT_SINGLE_SCATTERING;
  double dx = distanceToNearestAtmosphereBoundary(r, mu, rayRMuIntersectsGround) /
              SAMPLE_COUNT_SINGLE_SCATTERING;

  dvec2 currentDir = vec2(sqrt(1 - mu * mu), mu);
  vec2  sunDir     = vec2(sqrt(1 - muS * muS), muS);

  // Integration loop.
  vec3  moleculesSum     = vec3(0.0);
  vec3  aerosolsSum      = vec3(0.0);
  dvec3 transmittanceRay = dvec3(1.0);
  int   samples          = 0;

  dvec2 currentSamplePos = vec2(0.0, r);
  float currentR         = r;

  for (int i = 0; i <= SAMPLE_COUNT_SINGLE_SCATTERING; ++i) {
    // while (currentR <= TOP_RADIUS && ++samples < SAMPLE_COUNT_SINGLE_SCATTERING) {
    dvec2 nextSamplePos = currentSamplePos + currentDir * dx;

    vec2 currentSamplePosF = vec2(currentSamplePos);
    vec2 nextSamplePosF    = vec2(nextSamplePos);
    vec2 currentDirF       = vec2(currentDir);

    currentR        = length(currentSamplePosF);
    float nextR     = length(nextSamplePosF);
    float currentMu = clampCosine(dot(currentDirF, currentSamplePosF / currentR));
    float nextMuSD  = clampCosine(dot(sunDir, nextSamplePosF / nextR));

    vec3 transmittanceSun = getTransmittanceToSun(transmittanceTexture, nextR, nextMuSD);
    // vec3 transmittanceSun =
    //     getTransmittanceToTopAtmosphereBoundary(transmittanceTexture, nextR, nextMuSD);

    float moleculesOpticalDepth = float(dx) * getDensity(atmosphere.molecules.densityTextureV,
                                                  float(currentR) - BOTTOM_RADIUS);
    float aerosolsOpticalDepth  = float(dx) * getDensity(atmosphere.aerosols.densityTextureV,
                                                  float(currentR) - BOTTOM_RADIUS);
    float ozoneOpticalDepth =
        float(dx) * getDensity(atmosphere.ozone.densityTextureV, float(currentR) - BOTTOM_RADIUS);

    transmittanceRay *= exp(-(atmosphere.molecules.extinction * moleculesOpticalDepth +
                              atmosphere.aerosols.extinction * aerosolsOpticalDepth +
                              atmosphere.ozone.extinction * ozoneOpticalDepth));

    // transmittanceRay *= dvec3(getTransmittance(
    //     transmittanceTexture, currentR, currentMu, float(dx), rayRMuIntersectsGround));

    moleculesSum += transmittanceSun * vec3(transmittanceRay) *
                    getDensity(atmosphere.molecules.densityTextureV, nextR - BOTTOM_RADIUS);
    aerosolsSum += transmittanceSun * vec3(transmittanceRay) *
                   getDensity(atmosphere.aerosols.densityTextureV, nextR - BOTTOM_RADIUS);

    // If the ray intersects the ground, we have a problem: We do not have density information
    // in the underground, so the ray traversal will become undefined. Hence we clamp the ray
    // to the surface of the planet.
    if (currentR < BOTTOM_RADIUS) {
      rayRMuIntersectsGround = true;
    } else {
      currentDir = refractRay(currentSamplePos, currentDir, dx);
    }

    currentSamplePos = nextSamplePos;
  }

  molecules = moleculesSum * float(dx) * SOLAR_IRRADIANCE * atmosphere.molecules.scattering;
  aerosols  = aerosolsSum * float(dx) * SOLAR_IRRADIANCE * atmosphere.aerosols.scattering;
}

#else

void computeSingleScatteringIntegrand(AtmosphereComponents atmosphere,
    sampler2D transmittanceTexture, float r, float mu, float muS, float nu, float d,
    bool rayRMuIntersectsGround, out vec3 molecules, out vec3 aerosols) {
  float rD            = clampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  float muSD          = clampCosine((r * muS + d * nu) / rD);
  vec3  transmittance = getTransmittance(transmittanceTexture, r, mu, d, rayRMuIntersectsGround) *
                       getTransmittanceToSun(transmittanceTexture, rD, muSD);
  molecules = transmittance * getDensity(atmosphere.molecules.densityTextureV, rD - BOTTOM_RADIUS);
  aerosols  = transmittance * getDensity(atmosphere.aerosols.densityTextureV, rD - BOTTOM_RADIUS);
}

void computeSingleScattering(AtmosphereComponents atmosphere, sampler2D transmittanceTexture,
    float r, float mu, float muS, float nu, bool rayRMuIntersectsGround, out vec3 molecules,
    out vec3 aerosols) {

  // The integration step, i.e. the length of each integration interval.
  float dx = distanceToNearestAtmosphereBoundary(r, mu, rayRMuIntersectsGround) /
             float(SAMPLE_COUNT_SINGLE_SCATTERING);
  // Integration loop.
  vec3 moleculesSum = vec3(0.0);
  vec3 aerosolsSum  = vec3(0.0);
  for (int i = 0; i <= SAMPLE_COUNT_SINGLE_SCATTERING; ++i) {
    float d_i = float(i) * dx;
    // The Rayleigh and Mie single scattering at the current sample point.
    vec3 molecules_i;
    vec3 aerosols_i;
    computeSingleScatteringIntegrand(atmosphere, transmittanceTexture, r, mu, muS, nu, d_i,
        rayRMuIntersectsGround, molecules_i, aerosols_i);
    // Sample weight (from the trapezoidal rule).
    float weight_i = (i == 0 || i == SAMPLE_COUNT_SINGLE_SCATTERING) ? 0.5 : 1.0;
    moleculesSum += molecules_i * weight_i;
    aerosolsSum += aerosols_i * weight_i;
  }
  molecules = moleculesSum * dx * SOLAR_IRRADIANCE * atmosphere.molecules.scattering;
  aerosols  = aerosolsSum * dx * SOLAR_IRRADIANCE * atmosphere.aerosols.scattering;
}

#endif

vec3 phaseFunction(ScatteringComponent component, float nu) {
  float theta = acos(nu) / PI; // 0<->1
  return texture2D(uPhaseTexture, vec2(theta, component.phaseTextureV)).rgb;
}

// Single-Scattering Texture Precomputation --------------------------------------------------------

// The code below is used to store the single scattering (without the phase function applied) in a
// 4D lookup table.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#single_scattering_precomputation

// There is no functional difference to the original code.

void getRMuMuSNuFromScatteringTextureFragCoord(vec3 fragCoord, out float r, out float mu,
    out float muS, out float nu, out bool rayRMuIntersectsGround) {
  const vec4 SCATTERING_TEXTURE_SIZE = vec4(SCATTERING_TEXTURE_NU_SIZE - 1,
      SCATTERING_TEXTURE_MU_S_SIZE, SCATTERING_TEXTURE_MU_SIZE, SCATTERING_TEXTURE_R_SIZE);
  float      fragCoordNu             = floor(fragCoord.x / float(SCATTERING_TEXTURE_MU_S_SIZE));
  float      fragCoordMuS            = mod(fragCoord.x, float(SCATTERING_TEXTURE_MU_S_SIZE));
  vec4 uvwz = vec4(fragCoordNu, fragCoordMuS, fragCoord.y, fragCoord.z) / SCATTERING_TEXTURE_SIZE;
  getRMuMuSNuFromScatteringTextureUvwz(uvwz, r, mu, muS, nu, rayRMuIntersectsGround);
  // Clamp nu to its valid range of values, given mu and muS.
  nu = clamp(nu, mu * muS - sqrt((1.0 - mu * mu) * (1.0 - muS * muS)),
      mu * muS + sqrt((1.0 - mu * mu) * (1.0 - muS * muS)));
}

void computeSingleScatteringTexture(AtmosphereComponents atmosphere, sampler2D transmittanceTexture,
    vec3 fragCoord, out vec3 molecules, out vec3 aerosols) {
  float r;
  float mu;
  float muS;
  float nu;
  bool  rayRMuIntersectsGround;
  getRMuMuSNuFromScatteringTextureFragCoord(fragCoord, r, mu, muS, nu, rayRMuIntersectsGround);
  computeSingleScattering(atmosphere, transmittanceTexture, r, mu, muS, nu, rayRMuIntersectsGround,
      molecules, aerosols);
}

// Single-Scattering Texture Lookup ----------------------------------------------------------------

// The code below is used to retrieve the single-scattering values from the lookup tables.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#single_scattering_lookup

// There is no functional difference to the original code.

vec3 getScattering(sampler3D scatteringTexture, float r, float mu, float muS, float nu,
    bool rayRMuIntersectsGround) {
  vec4  uvwz      = getScatteringTextureUvwzFromRMuMuSNu(r, mu, muS, nu, rayRMuIntersectsGround);
  float texCoordX = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
  float texX      = floor(texCoordX);
  float lerp      = texCoordX - texX;
  vec3  uvw0      = vec3((texX + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  vec3  uvw1      = vec3((texX + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  return vec3(
      texture(scatteringTexture, uvw0) * (1.0 - lerp) + texture(scatteringTexture, uvw1) * lerp);
}

vec3 getScattering(AtmosphereComponents atmosphere, sampler3D singleMoleculesScatteringTexture,
    sampler3D singleAerosolsScatteringTexture, sampler3D multipleScatteringTexture, float r,
    float mu, float muS, float nu, bool rayRMuIntersectsGround, int scatteringOrder) {
  if (scatteringOrder == 1) {
    vec3 molecules =
        getScattering(singleMoleculesScatteringTexture, r, mu, muS, nu, rayRMuIntersectsGround);
    vec3 aerosols =
        getScattering(singleAerosolsScatteringTexture, r, mu, muS, nu, rayRMuIntersectsGround);
    return molecules * phaseFunction(atmosphere.molecules, nu) +
           aerosols * phaseFunction(atmosphere.aerosols, nu);
  } else {
    return getScattering(multipleScatteringTexture, r, mu, muS, nu, rayRMuIntersectsGround);
  }
}

// Multiple-Scattering Computation -----------------------------------------------------------------

// The code below is used to compute the amount of light scattered after more than one bounces in
// the atmosphere.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#multipleScattering

// There is no functional difference to the original code.

vec3 getIrradiance(sampler2D irradianceTexture, float r, float muS);

vec3 computeScatteringDensity(AtmosphereComponents atmosphere, sampler2D transmittanceTexture,
    sampler3D singleMoleculesScatteringTexture, sampler3D singleAerosolsScatteringTexture,
    sampler3D multipleScatteringTexture, sampler2D irradianceTexture, float r, float mu, float muS,
    float nu, int scatteringOrder) {

  // Compute unit direction vectors for the zenith, the view direction omega and and the sun
  // direction omegaS, such that the cosine of the view-zenith angle is mu, the cosine of the
  // sun-zenith angle is muS, and the cosine of the view-sun angle is nu. The goal is to simplify
  // computations below.
  vec3  zenithDirection = vec3(0.0, 0.0, 1.0);
  vec3  omega           = vec3(sqrt(1.0 - mu * mu), 0.0, mu);
  float sunDirX         = omega.x == 0.0 ? 0.0 : (nu - mu * muS) / omega.x;
  float sunDirY         = sqrt(max(1.0 - sunDirX * sunDirX - muS * muS, 0.0));
  vec3  omegaS          = vec3(sunDirX, sunDirY, muS);

  const float dPhi              = PI / float(SAMPLE_COUNT_SCATTERING_DENSITY);
  const float dTheta            = PI / float(SAMPLE_COUNT_SCATTERING_DENSITY);
  vec3        moleculesAerosols = vec3(0.0);

  // Nested loops for the integral over all the incident directions omega_i.
  for (int l = 0; l < SAMPLE_COUNT_SCATTERING_DENSITY; ++l) {
    float theta                     = (float(l) + 0.5) * dTheta;
    float cosTheta                  = cos(theta);
    float sinTheta                  = sin(theta);
    bool  rayRThetaIntersectsGround = rayIntersectsGround(r, cosTheta);

    // The distance and transmittance to the ground only depend on theta, so we can compute them in
    // the outer loop for efficiency.
    float distanceToGround      = 0.0;
    vec3  transmittanceToGround = vec3(0.0);
    vec3  groundAlbedo          = vec3(0.0);
    if (rayRThetaIntersectsGround) {
      distanceToGround      = distanceToBottomAtmosphereBoundary(r, cosTheta);
      transmittanceToGround = getTransmittance(
          transmittanceTexture, r, cosTheta, distanceToGround, true /* ray_intersects_ground */);
      groundAlbedo = GROUND_ALBEDO;
    }

    for (int m = 0; m < 2 * SAMPLE_COUNT_SCATTERING_DENSITY; ++m) {
      float phi      = (float(m) + 0.5) * dPhi;
      vec3  omega_i  = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
      float domega_i = dTheta * dPhi * sin(theta);

      // The radiance L_i arriving from direction omega_i after n-1 bounces is the sum of a term
      // given by the precomputed scattering texture for the (n-1)-th order:
      float nu1              = dot(omegaS, omega_i);
      vec3  incidentRadiance = getScattering(atmosphere, singleMoleculesScatteringTexture,
           singleAerosolsScatteringTexture, multipleScatteringTexture, r, omega_i.z, muS, nu1,
           rayRThetaIntersectsGround, scatteringOrder - 1);

      // and of the contribution from the light paths with n-1 bounces and whose last bounce is on
      // the ground. This contribution is the product of the transmittance to the ground, the ground
      // albedo, the ground BRDF, and the irradiance received on the ground after n-2 bounces.
      vec3 groundNormal = normalize(zenithDirection * r + omega_i * distanceToGround);
      vec3 groundIrradiance =
          getIrradiance(irradianceTexture, BOTTOM_RADIUS, dot(groundNormal, omegaS));
      incidentRadiance += transmittanceToGround * groundAlbedo * (1.0 / PI) * groundIrradiance;

      // The radiance finally scattered from direction omega_i towards direction -omega is the
      // product of the incident radiance, the scattering coefficient, and the phase function for
      // directions omega and omega_i (all this summed over all particle types, i.e. Rayleigh and
      // Mie).
      float nu2              = dot(omega, omega_i);
      float moleculesDensity = getDensity(atmosphere.molecules.densityTextureV, r - BOTTOM_RADIUS);
      float aerosolsDensity  = getDensity(atmosphere.aerosols.densityTextureV, r - BOTTOM_RADIUS);
      moleculesAerosols += incidentRadiance *
                           (atmosphere.molecules.scattering * moleculesDensity *
                                   phaseFunction(atmosphere.molecules, nu2) +
                               atmosphere.aerosols.scattering * aerosolsDensity *
                                   phaseFunction(atmosphere.aerosols, nu2)) *
                           domega_i;
    }
  }
  return moleculesAerosols;
}

vec3 computeMultipleScattering(sampler2D transmittanceTexture, sampler3D scatteringDensityTexture,
    float r, float mu, float muS, float nu, bool rayRMuIntersectsGround) {

  // The integration step, i.e. the length of each integration interval.
  float dx = distanceToNearestAtmosphereBoundary(r, mu, rayRMuIntersectsGround) /
             float(SAMPLE_COUNT_MULTI_SCATTERING);
  // Integration loop.
  vec3 moleculesAerosolsSum = vec3(0.0);
  for (int i = 0; i <= SAMPLE_COUNT_MULTI_SCATTERING; ++i) {
    float d_i = float(i) * dx;

    // The r, mu and muS parameters at the current integration point (see the single scattering
    // section for a detailed explanation).
    float r_i   = clampRadius(sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r));
    float mu_i  = clampCosine((r * mu + d_i) / r_i);
    float muS_i = clampCosine((r * muS + d_i * nu) / r_i);

    // The Rayleigh and Mie multiple scattering at the current sample point.
    vec3 moleculesAerosols_i =
        getScattering(scatteringDensityTexture, r_i, mu_i, muS_i, nu, rayRMuIntersectsGround) *
        getTransmittance(transmittanceTexture, r, mu, d_i, rayRMuIntersectsGround) * dx;
    // Sample weight (from the trapezoidal rule).
    float weight_i = (i == 0 || i == SAMPLE_COUNT_MULTI_SCATTERING) ? 0.5 : 1.0;
    moleculesAerosolsSum += moleculesAerosols_i * weight_i;
  }
  return moleculesAerosolsSum;
}

// Multiple-Scattering Texture Precomputation ------------------------------------------------------

// The code below is used to store the multiple scattering (with the phase function applied) in a
// 4D lookup table.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#multiple_scattering_precomputation

// There is no functional difference to the original code.

vec3 computeScatteringDensityTexture(AtmosphereComponents atmosphere,
    sampler2D transmittanceTexture, sampler3D singleMoleculesScatteringTexture,
    sampler3D singleAerosolsScatteringTexture, sampler3D multipleScatteringTexture,
    sampler2D irradianceTexture, vec3 fragCoord, int scatteringOrder) {
  float r;
  float mu;
  float muS;
  float nu;
  bool  rayRMuIntersectsGround;
  getRMuMuSNuFromScatteringTextureFragCoord(fragCoord, r, mu, muS, nu, rayRMuIntersectsGround);
  return computeScatteringDensity(atmosphere, transmittanceTexture,
      singleMoleculesScatteringTexture, singleAerosolsScatteringTexture, multipleScatteringTexture,
      irradianceTexture, r, mu, muS, nu, scatteringOrder);
}

vec3 computeMultipleScatteringTexture(sampler2D transmittanceTexture,
    sampler3D scatteringDensityTexture, vec3 fragCoord, out float nu) {
  float r;
  float mu;
  float muS;
  bool  rayRMuIntersectsGround;
  getRMuMuSNuFromScatteringTextureFragCoord(fragCoord, r, mu, muS, nu, rayRMuIntersectsGround);
  return computeMultipleScattering(
      transmittanceTexture, scatteringDensityTexture, r, mu, muS, nu, rayRMuIntersectsGround);
}

// Compute Irradiance ------------------------------------------------------------------------------

// The code below is used to compute the irradiance received from the Sun and from the sky at a
// given altitude.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#irradiance

// There is no functional difference to the original code.

vec3 computeDirectIrradiance(sampler2D transmittanceTexture, float r, float muS) {

  float alphaS = SUN_ANGULAR_RADIUS;
  // Approximate average of the cosine factor muS over the visible fraction of
  // the Sun disc.
  float averageCosineFactor =
      muS < -alphaS ? 0.0 : (muS > alphaS ? muS : (muS + alphaS) * (muS + alphaS) / (4.0 * alphaS));

  return SOLAR_IRRADIANCE * getTransmittanceToTopAtmosphereBoundary(transmittanceTexture, r, muS) *
         averageCosineFactor;
}

vec3 computeIndirectIrradiance(AtmosphereComponents atmosphere,
    sampler3D singleMoleculesScatteringTexture, sampler3D singleAerosolsScatteringTexture,
    sampler3D multipleScatteringTexture, float r, float muS, int scatteringOrder) {

  const float dPhi   = PI / float(SAMPLE_COUNT_INDIRECT_IRRADIANCE);
  const float dTheta = PI / float(SAMPLE_COUNT_INDIRECT_IRRADIANCE);

  vec3 result = vec3(0.0);
  vec3 omegaS = vec3(sqrt(1.0 - muS * muS), 0.0, muS);
  for (int j = 0; j < SAMPLE_COUNT_INDIRECT_IRRADIANCE / 2; ++j) {
    float theta = (float(j) + 0.5) * dTheta;
    for (int i = 0; i < 2 * SAMPLE_COUNT_INDIRECT_IRRADIANCE; ++i) {
      float phi    = (float(i) + 0.5) * dPhi;
      vec3  omega  = vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
      float domega = dTheta * dPhi * sin(theta);

      float nu = dot(omega, omegaS);
      result += getScattering(atmosphere, singleMoleculesScatteringTexture,
                    singleAerosolsScatteringTexture, multipleScatteringTexture, r, omega.z, muS, nu,
                    false /* rayRThetaIntersectsGround */, scatteringOrder) *
                omega.z * domega;
    }
  }
  return result;
}

// Irradiance-Texture Precomputation ---------------------------------------------------------------

// The code below is used to store the direct and indirect irradiance received at any altitude in 2D
// lookup tables.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#irradiance_precomputation

// There is no functional difference to the original code.

vec3 computeDirectIrradianceTexture(sampler2D transmittanceTexture, vec2 fragCoord) {
  float r;
  float muS;
  getRMuSFromIrradianceTextureUv(
      fragCoord / vec2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT), r, muS);
  return computeDirectIrradiance(transmittanceTexture, r, muS);
}

vec3 computeIndirectIrradianceTexture(AtmosphereComponents atmosphere,
    sampler3D singleMoleculesScatteringTexture, sampler3D singleAerosolsScatteringTexture,
    sampler3D multipleScatteringTexture, vec2 fragCoord, int scatteringOrder) {
  float r;
  float muS;
  getRMuSFromIrradianceTextureUv(
      fragCoord / vec2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT), r, muS);
  return computeIndirectIrradiance(atmosphere, singleMoleculesScatteringTexture,
      singleAerosolsScatteringTexture, multipleScatteringTexture, r, muS, scatteringOrder);
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
