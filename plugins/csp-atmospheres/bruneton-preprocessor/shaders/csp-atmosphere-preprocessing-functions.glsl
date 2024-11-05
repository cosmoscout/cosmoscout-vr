////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-FileCopyrightText: 2008 INRIA
// SPDX-License-Identifier: BSD-3-Clause

#line 11

// This file is roughly based on the original implementation by Eric Bruneton:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl

// While implementing the atmospheric model into CosmoScout VR, we have extended this model in
// several ways. There are two fundamental changes:
//   1. The phase functions for aerosols and molecules as well as their density distributions are
//      now sampled from textures.
//   2. The index of refraction of the atmosphere is now passed to the shader and if
//      COMPUTE_REFRACTION is set, rays are refracted accordingly.

// Below, we will indicate for each group of function whether something important has been changed
// with respect to the original implementation and add a link to the original explanation.

// Refraction Computation --------------------------------------------------------------------------

// This part is new. The wavelength-dependent INDEX_OF_REFRACTION is passed to the shader. According
// to the Gladsstone-Dale relation, the refractive index is proportional to the density of the air
// molecules. The density of the air molecules and aerosols is sampled from a texture.

// This returns the relative density in  [0...1] of a component at a given altitude (in meters). The
// density texture contains the density of the air molecules, aerosols, or ozone molecules at
// different v coordinates, so this needs to be passed as an argument.
float getDensity(float densityTextureV, float altitude) {
  float u = clamp(altitude / (TOP_RADIUS - BOTTOM_RADIUS), 0.0, 1.0);
  return texture(uDensityTexture, vec2(u, densityTextureV)).r;
}

// Using acos is not very stable for small angles. This function is used to compute small angles
// between two normalized vectors in a more stable way.
float angleBetweenVectors(vec2 u, vec2 v) {
  return 2.0 * asin(0.5 * length(u - v));
}

#if COMPUTE_REFRACTION

// Returns IoR - 1.0 at the given altitude. Here, single floating point precision is sufficient.
float getRefractiveIndexMinusOne(float altitude) {
  return INDEX_OF_REFRACTION * getDensity(ATMOSPHERE.molecules.densityTextureV, altitude);
}

// Returns the IoR at the given altitude. To properly represent very small changes in the refractive
// index, we use double precision here.
double getRefractiveIndex(float altitude) {
  return 1.0lf + double(getRefractiveIndexMinusOne(altitude));
}

// Returns the gradient length of the refractive index at the given altitude.
float getIoRGradientLength(float altitude, float dh) {
  return (getRefractiveIndexMinusOne(altitude + dh) - getRefractiveIndexMinusOne(altitude)) / dh;
}

// Refracts the ray according to "Visualizing sunsets through inhomogeneous atmospheres" by
// Seron et al. (2004) using a fixed step size.
dvec2 refractRaySeron(dvec2 origin, dvec2 dir, double dx) {
  float  altitude        = max(0, float(length(origin) - BOTTOM_RADIUS));
  double refractiveIndex = getRefractiveIndex(altitude);
  double gradientLength  = getIoRGradientLength(altitude, 100);
  dvec2  dn              = normalize(origin) * gradientLength;
  return normalize(refractiveIndex * dir + dn * dx);
}

// Refracts the ray according to "Comment on 'Improved ray tracing air mass numbers model' by van
// der Werf (2008). Due to the involved trigonometry, have to use single precision in some places.
dvec2 refractRayWerf(dvec2 origin, dvec2 dir, double dx) {

  double RplusH = length(origin);
  float  h      = float(max(0, RplusH - BOTTOM_RADIUS));

  // This could be computed using cos(PI / 2.0 - acos(dot(origin / RplusH, dir)), however it can
  // be simplified to the following using trigonometric identities.
  double cosBeta = sqrt(1.0 - pow(float(dot(origin / RplusH, dir)), 2.0));

  double curvature = cosBeta / getRefractiveIndex(h) * getIoRGradientLength(h, 100);
  double deltaBeta = dx * (cosBeta / RplusH + curvature);
  double deltaPhi  = cosBeta * dx / RplusH;
  float  diff      = float(deltaBeta - deltaPhi);

  // Rotate dir by diff.
  return dvec2(dir.x * cos(diff) - dir.y * sin(diff), dir.x * sin(diff) + dir.y * cos(diff));
}

// Use this to switch between the two refraction methods.
#define refractRay refractRaySeron

// The Euler method just advances the ray in the direction of the refracted ray.
void rayStepEuler(inout dvec2 origin, inout dvec2 dir, double dx) {
  dir = refractRay(origin, dir, dx);
  origin += dir * dx;
}

// The Runge-Kutta 4 method is more accurate but also more expensive.
void rayStepRK4(inout dvec2 origin, inout dvec2 dir, double dx) {
  dvec2 k1 = refractRay(origin, dir, dx);
  dvec2 k2 = refractRay(origin + k1 * dx / 2.0, dir, dx);
  dvec2 k3 = refractRay(origin + k2 * dx / 2.0, dir, dx);
  dvec2 k4 = refractRay(origin + k3 * dx, dir, dx);
  dir      = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
  origin += dir * dx;
}

// Use this to switch between the two ray stepping methods.
#define rayStep rayStepRK4

#endif // COMPUTE_REFRACTION

// Transmittance Computation -----------------------------------------------------------------------

// The code below is used to comute the optical depth (or transmittance) from any point in the
// atmosphere towards the top atmosphere boundary.

// The first functional difference is that the density of the air molecules, aerosols, and ozone
// molecules is now sampled from a texture (in getDensity()) instead of analytically computed.
// In addition, we have two variants of the computeOpticalLengthToTopAtmosphereBoundary method.
// If COMPUTE_REFRACTION is set, the ray is bent according to the refractive index of the air
// molecules. If not, we use the original implementation by Eric Bruneton with evenly spaced samples
// along the ray.

#if COMPUTE_REFRACTION

// If our ray hit the surface of the planet due to refraction, the optical depth should be infinite.
// However, returning a very high optical depth is not a good solution, because the transmittance
// computation in getTransmittance() in common.glsl relies on the fact that the ray does not
// intersect the planet in either the forward or backward direction.
// As a solution, we store the contact radius (which is the altitude of the closest approach of the
// ray to the planet and which is negative if the ray intersects the planet) in the RayInfo struct.
// This value is then used to determine if the ray actually hit the planet.
// The thetaDeviation is the angle between the original ray direction and the refracted ray
// direction. This is used to displace astronomical bodies in the sky.
struct RayInfo {
  float opticalDepth;
  float thetaDeviation;
  float contactRadius;
};

RayInfo computeOpticalLengthToTopAtmosphereBoundary(float densityTextureV, float r, float mu) {

  double dx          = STEP_SIZE_OPTICAL_DEPTH;
  dvec2  startRayDir = vec2(sqrt(1 - mu * mu), mu);

  RayInfo result;
  result.opticalDepth   = 0.0;
  result.thetaDeviation = 0.0;
  result.contactRadius  = r - BOTTOM_RADIUS;

  dvec2  samplePos    = vec2(0.0, r);
  double sampleRadius = r;
  dvec2  currentDir   = dvec2(sqrt(1 - mu * mu), mu);

  while (sampleRadius <= TOP_RADIUS + 10) {
    sampleRadius = length(samplePos);

    float altitude = float(sampleRadius) - BOTTOM_RADIUS;
    result.opticalDepth += getDensity(densityTextureV, altitude);
    result.contactRadius = min(result.contactRadius, altitude);

    rayStep(samplePos, currentDir, dx);
  }

  result.thetaDeviation = angleBetweenVectors(vec2(startRayDir), vec2(currentDir));
  result.opticalDepth *= float(dx);

  return result;
}

#else

// If no refraction is computed, the optical depth is simply the integral of the density along the
// ray. We use the original implementation by Eric Bruneton with evenly spaced samples along the
// ray. An explanation of this method is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#transmittance
struct RayInfo {
  float opticalDepth;
};

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

// An explanation of the following method is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#transmittance_precomputation

// The only difference to the original implementation is that the method also returns the theta
// deviation and the contact radius if COMPUTE_REFRACTION is set.

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
// If refraction is enabled, we cannot use the original implementation by Eric Bruneton, because the
// ray is bent.

#if COMPUTE_REFRACTION

// Double precision variant of the same method defined in common.glsl.
double clampCosine(double mu) {
  return clamp(mu, -1.0, 1.0);
}

// Returns the transmittance of a ray segment of length dx. The transmittance is based on the
// atmospheric density at the given planet-center distance and the extinction coefficients of the
// air molecules, aerosols, and ozone.
vec3 getTransmittanceForRaySegment(AtmosphereComponents atmosphere, float r, float dx) {
  float altitude         = r - BOTTOM_RADIUS;
  float moleculesDensity = getDensity(atmosphere.molecules.densityTextureV, altitude);
  float aerosolsDensity  = getDensity(atmosphere.aerosols.densityTextureV, altitude);
  float ozoneDensity     = getDensity(atmosphere.ozone.densityTextureV, altitude);

  return exp(-dx * (atmosphere.molecules.extinction * moleculesDensity +
                       atmosphere.aerosols.extinction * aerosolsDensity +
                       atmosphere.ozone.extinction * ozoneDensity));
}

// The direction to the sun is encoded using the cosines of the zenith angle muS and the cosine to
// the ray direction nu. We reconstruct the 3D direction vector from these two values.
vec3 getSunDirection(float mu, float muS, float nu) {
  float rayDirX = safeSqrt(1 - mu * mu);
  float rayDirY = mu;

  float sunDirX = (nu - rayDirY * muS) / (rayDirX + 1e-20);
  float sunDirY = muS;
  float sunDirZ = safeSqrt(1 - sunDirX * sunDirX - sunDirY * sunDirY);

  return vec3(sunDirX, sunDirY, sunDirZ);
}

// As for the transmittance, the single-scattering computation is different if refraction is
// incorporated. The ray is bent, so we cannot simply accumulate the scattering along a straight
// line.
// We want to keep the code as close to the original implementation as possible. Therefore, we
// did not change the signature of the method. However, we now trace a 2D ray in the atmosphere and
// therefore need the 3D direction to the sun. The method above is used to reconstruct this.
// If we were to remove the code without refraction, we would trace the rays in 2D and could pass
// the sun direction directly to the method.
void computeSingleScattering(AtmosphereComponents atmosphere, sampler2D transmittanceTexture,
    float r, float mu, float muS, float nu, bool rayRMuIntersectsGround, out vec3 molecules,
    out vec3 aerosols) {

  dvec2 currentDir = vec2(sqrt(1 - mu * mu), mu);
  vec3  sunDir     = getSunDirection(mu, muS, nu);

  vec3  moleculesSum     = vec3(0.0);
  vec3  aerosolsSum      = vec3(0.0);
  dvec3 transmittanceRay = dvec3(1.0);

  dvec2  samplePos    = vec2(0.0, r);
  double sampleRadius = r;
  bool   hitGround    = false;

  while (sampleRadius <= TOP_RADIUS + 10 && !hitGround) {
    double dx            = STEP_SIZE_SINGLE_SCATTERING;
    dvec2  nextSamplePos = samplePos + currentDir * dx;
    double nextR         = length(nextSamplePos);

    // If the segment intersects the ground, we shorten the segment to the intersection point.
    if (nextR < BOTTOM_RADIUS) {
      dx            = dx * (sampleRadius - BOTTOM_RADIUS) / (sampleRadius - nextR);
      nextSamplePos = samplePos + currentDir * dx;
      nextR         = BOTTOM_RADIUS;
      hitGround     = true;
    }

    float muSD = clampCosine(dot(sunDir, vec3(nextSamplePos, 0.0)) / float(nextR));

    vec3 transmittanceSun = getTransmittanceToSun(transmittanceTexture, float(nextR), muSD);
    transmittanceRay *= getTransmittanceForRaySegment(atmosphere, float(sampleRadius), float(dx));

    float nextAltitude     = float(nextR) - BOTTOM_RADIUS;
    float moleculesDensity = getDensity(atmosphere.molecules.densityTextureV, nextAltitude);
    float aerosolsDensity  = getDensity(atmosphere.aerosols.densityTextureV, nextAltitude);
    moleculesSum += transmittanceSun * vec3(transmittanceRay) * moleculesDensity * float(dx);
    aerosolsSum += transmittanceSun * vec3(transmittanceRay) * aerosolsDensity * float(dx);

    rayStep(samplePos, currentDir, dx);
    sampleRadius = length(samplePos);
  }

  molecules = moleculesSum * SOLAR_IRRADIANCE * atmosphere.molecules.scattering;
  aerosols  = aerosolsSum * SOLAR_IRRADIANCE * atmosphere.aerosols.scattering;
}

#else

// If no refraction is computed, the single-scattering computation is the same as in the original
// implementation by Eric Bruneton. The ray is not bent, so we can accumulate the scattering along a
// straight line.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#single_scattering

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

float distanceToNearestAtmosphereBoundary(float r, float mu, bool rayRMuIntersectsGround) {
  if (rayRMuIntersectsGround) {
    return distanceToBottomAtmosphereBoundary(r, mu);
  } else {
    return distanceToTopAtmosphereBoundary(r, mu);
  }
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

// The RayleighPhaseFunction() and MiePhaseFunction() have been removed and replaced by a generic
// phaseFunction() which samples the phase function from a texture.
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
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#multiple_scattering

// Similar to the single-scattering computation, the multiple-scattering computation is different if
// refraction is incorporated. The ray is bent, so we cannot simply accumulate the scattering along
// a straight line.

// All other methods are basically the same as in the original implementation by Eric Bruneton.

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

#if COMPUTE_REFRACTION

// As for the single-scattering computation, the multiple-scattering computation is different if
// refraction is incorporated. We trace the rays in 2D and use a reconstructed 3D sun light
// direction.

vec3 computeMultipleScattering(AtmosphereComponents atmosphere, sampler2D transmittanceTexture,
    sampler3D scatteringDensityTexture, float r, float mu, float muS, float nu,
    bool rayRMuIntersectsGround) {

  dvec2 currentDir = vec2(sqrt(1 - mu * mu), mu);
  vec3  sunDir     = getSunDirection(mu, muS, nu);

  vec3  moleculesAerosolsSum = vec3(0.0);
  dvec3 transmittanceRay     = dvec3(1.0);

  dvec2  samplePos    = vec2(0.0, r);
  double sampleRadius = r;
  bool   hitGround    = false;

  while (sampleRadius <= TOP_RADIUS + 10 && !hitGround) {
    double dx            = STEP_SIZE_MULTI_SCATTERING;
    dvec2  nextSamplePos = samplePos + currentDir * dx;
    double nextR         = length(nextSamplePos);

    // If the segment intersects the ground, we shorten the segment to the intersection point.
    if (nextR < BOTTOM_RADIUS) {
      dx            = dx * (sampleRadius - BOTTOM_RADIUS) / (sampleRadius - nextR);
      nextSamplePos = samplePos + currentDir * dx;
      nextR         = BOTTOM_RADIUS;
      hitGround     = true;
    }

    double currentMu  = clampCosine(dot(nextSamplePos / nextR, currentDir));
    double currentMuS = clampCosine(dot(dvec3(nextSamplePos, 0.0) / nextR, sunDir));
    double currentNu  = clampCosine(dot(dvec3(currentDir, 0.0), sunDir));

    transmittanceRay *= getTransmittanceForRaySegment(atmosphere, float(nextR), float(dx));

    moleculesAerosolsSum += getScattering(scatteringDensityTexture, float(nextR), float(currentMu),
                                float(currentMuS), float(currentNu), rayRMuIntersectsGround) *
                            vec3(transmittanceRay) * float(dx);

    rayStep(samplePos, currentDir, dx);
    sampleRadius = length(samplePos);
  }

  return moleculesAerosolsSum;
}

#else

// See the second step of
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#multiple_scattering

vec3 computeMultipleScattering(AtmosphereComponents atmosphere, sampler2D transmittanceTexture,
    sampler3D scatteringDensityTexture, float r, float mu, float muS, float nu,
    bool rayRMuIntersectsGround) {

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

#endif

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

vec3 computeMultipleScatteringTexture(AtmosphereComponents atmosphere,
    sampler2D transmittanceTexture, sampler3D scatteringDensityTexture, vec3 fragCoord,
    out float nu) {
  float r;
  float mu;
  float muS;
  bool  rayRMuIntersectsGround;
  getRMuMuSNuFromScatteringTextureFragCoord(fragCoord, r, mu, muS, nu, rayRMuIntersectsGround);
  return computeMultipleScattering(atmosphere, transmittanceTexture, scatteringDensityTexture, r,
      mu, muS, nu, rayRMuIntersectsGround);
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
