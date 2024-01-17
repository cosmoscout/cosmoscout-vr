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

float ClampCosine(float mu) {
  return clamp(mu, -1.0, 1.0);
}

float ClampDistance(float d) {
  return max(d, 0.0);
}

float ClampRadius(float r) {
  return clamp(r, BOTTOM_RADIUS, TOP_RADIUS);
}

float SafeSqrt(float a) {
  return sqrt(max(a, 0.0));
}

// Transmittance Computation -----------------------------------------------------------------------

// The code below is used to comute the optical depth (or transmittance) from any point in the
// atmosphere towards the top atmosphere boundary.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#transmittance

// The only functional difference is that the density of the air molecules, aerosols, and ozone
// molecules is now sampled from a texture (in GetDensity()) instead of analytically computed.

float DistanceToTopAtmosphereBoundary(float r, float mu) {
  float discriminant = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS;
  return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

float DistanceToBottomAtmosphereBoundary(float r, float mu) {
  float discriminant = r * r * (mu * mu - 1.0) + BOTTOM_RADIUS * BOTTOM_RADIUS;
  return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

bool RayIntersectsGround(float r, float mu) {
  return mu < 0.0 && r * r * (mu * mu - 1.0) + BOTTOM_RADIUS * BOTTOM_RADIUS >= 0.0;
}

float GetDensity(float densityTextureV, float altitude) {
  float u = clamp(altitude / (TOP_RADIUS - BOTTOM_RADIUS), 0.0, 1.0);
  return texture(density_texture, vec2(u, densityTextureV)).r;
}

float ComputeOpticalLengthToTopAtmosphereBoundary(float densityTextureV, float r, float mu) {
  float dx     = DistanceToTopAtmosphereBoundary(r, mu) / float(SAMPLE_COUNT_OPTICAL_DEPTH);
  float result = 0.0;
  for (int i = 0; i <= SAMPLE_COUNT_OPTICAL_DEPTH; ++i) {
    float d_i      = float(i) * dx;
    float r_i      = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
    float y_i      = GetDensity(densityTextureV, r_i - BOTTOM_RADIUS);
    float weight_i = i == 0 || i == SAMPLE_COUNT_OPTICAL_DEPTH ? 0.5 : 1.0;
    result += y_i * weight_i * dx;
  }
  return result;
}

vec3 ComputeTransmittanceToTopAtmosphereBoundary(
    AtmosphereComponents atmosphere, float r, float mu) {
  return exp(-(atmosphere.molecules.extinction * ComputeOpticalLengthToTopAtmosphereBoundary(
                                                     atmosphere.molecules.densityTextureV, r, mu) +
               atmosphere.aerosols.extinction * ComputeOpticalLengthToTopAtmosphereBoundary(
                                                    atmosphere.aerosols.densityTextureV, r, mu) +
               atmosphere.ozone.extinction * ComputeOpticalLengthToTopAtmosphereBoundary(
                                                 atmosphere.ozone.densityTextureV, r, mu)));
}

// Transmittance Texture Pre-Computation -----------------------------------------------------------

// The code below is used to store the pre-computed transmittance values in a 2D lookup table.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#transmittance_precomputation

// There is no functional difference to the original code.

float GetTextureCoordFromUnitRange(float x, int texture_size) {
  return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

float GetUnitRangeFromTextureCoord(float u, int texture_size) {
  return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}

vec2 GetTransmittanceTextureUvFromRMu(float r, float mu) {
  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon.
  float rho = SafeSqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
  // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
  float d     = DistanceToTopAtmosphereBoundary(r, mu);
  float d_min = TOP_RADIUS - r;
  float d_max = rho + H;
  float x_mu  = (d - d_min) / (d_max - d_min);
  float x_r   = rho / H;
  return vec2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
      GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}

void GetRMuFromTransmittanceTextureUv(vec2 uv, out float r, out float mu) {
  float x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
  float x_r  = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon, from which we can compute r:
  float rho = H * x_r;
  r         = sqrt(rho * rho + BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
  // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon) -
  // from which we can recover mu:
  float d_min = TOP_RADIUS - r;
  float d_max = rho + H;
  float d     = d_min + x_mu * (d_max - d_min);
  mu          = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
  mu          = ClampCosine(mu);
}

vec3 ComputeTransmittanceToTopAtmosphereBoundaryTexture(
    AtmosphereComponents atmosphere, vec2 frag_coord) {
  const vec2 TRANSMITTANCE_TEXTURE_SIZE =
      vec2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
  float r;
  float mu;
  GetRMuFromTransmittanceTextureUv(frag_coord / TRANSMITTANCE_TEXTURE_SIZE, r, mu);
  return ComputeTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
}

// Transmittance Texture Lookup --------------------------------------------------------------------

// The code below is used to retrieve the transmittance values from the lookup tables.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#transmittance_lookup

// There is no functional difference to the original code.

vec3 GetTransmittanceToTopAtmosphereBoundary(sampler2D transmittance_texture, float r, float mu) {
  vec2 uv = GetTransmittanceTextureUvFromRMu(r, mu);
  return vec3(texture(transmittance_texture, uv));
}

vec3 GetTransmittance(
    sampler2D transmittance_texture, float r, float mu, float d, bool ray_r_mu_intersects_ground) {

  float r_d  = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  float mu_d = ClampCosine((r * mu + d) / r_d);

  if (ray_r_mu_intersects_ground) {
    return min(GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r_d, -mu_d) /
                   GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, -mu),
        vec3(1.0));
  } else {
    return min(GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu) /
                   GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r_d, mu_d),
        vec3(1.0));
  }
}

vec3 GetTransmittanceToSun(sampler2D transmittance_texture, float r, float mu_s) {
  float sin_theta_h = BOTTOM_RADIUS / r;
  float cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
  return GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu_s) *
         smoothstep(-sin_theta_h * SUN_ANGULAR_RADIUS, sin_theta_h * SUN_ANGULAR_RADIUS,
             mu_s - cos_theta_h);
}

// Single-Scattering Computation -------------------------------------------------------------------

// The code below is used to compute the amount of light scattered into a specific direction during
// a single scattering event for air molecules and aerosols.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#single_scattering

// Most of the methods below are functionality-wise identical to the original implementation. The
// only difference is that the RayleighPhaseFunction() and MiePhaseFunction() have been removed and
// replaced by a generic PhaseFunction() which samples the phase function from a texture.

void ComputeSingleScatteringIntegrand(AtmosphereComponents atmosphere,
    sampler2D transmittance_texture, float r, float mu, float mu_s, float nu, float d,
    bool ray_r_mu_intersects_ground, out vec3 molecules, out vec3 aerosols) {
  float r_d    = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  float mu_s_d = ClampCosine((r * mu_s + d * nu) / r_d);
  vec3  transmittance =
      GetTransmittance(transmittance_texture, r, mu, d, ray_r_mu_intersects_ground) *
      GetTransmittanceToSun(transmittance_texture, r_d, mu_s_d);
  molecules = transmittance * GetDensity(atmosphere.molecules.densityTextureV, r_d - BOTTOM_RADIUS);
  aerosols  = transmittance * GetDensity(atmosphere.aerosols.densityTextureV, r_d - BOTTOM_RADIUS);
}

float DistanceToNearestAtmosphereBoundary(float r, float mu, bool ray_r_mu_intersects_ground) {
  if (ray_r_mu_intersects_ground) {
    return DistanceToBottomAtmosphereBoundary(r, mu);
  } else {
    return DistanceToTopAtmosphereBoundary(r, mu);
  }
}

void ComputeSingleScattering(AtmosphereComponents atmosphere, sampler2D transmittance_texture,
    float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground, out vec3 molecules,
    out vec3 aerosols) {

  // The integration step, i.e. the length of each integration interval.
  float dx = DistanceToNearestAtmosphereBoundary(r, mu, ray_r_mu_intersects_ground) /
             float(SAMPLE_COUNT_SINGLE_SCATTERING);
  // Integration loop.
  vec3 molecules_sum = vec3(0.0);
  vec3 aerosols_sum  = vec3(0.0);
  for (int i = 0; i <= SAMPLE_COUNT_SINGLE_SCATTERING; ++i) {
    float d_i = float(i) * dx;
    // The Rayleigh and Mie single scattering at the current sample point.
    vec3 molecules_i;
    vec3 aerosols_i;
    ComputeSingleScatteringIntegrand(atmosphere, transmittance_texture, r, mu, mu_s, nu, d_i,
        ray_r_mu_intersects_ground, molecules_i, aerosols_i);
    // Sample weight (from the trapezoidal rule).
    float weight_i = (i == 0 || i == SAMPLE_COUNT_SINGLE_SCATTERING) ? 0.5 : 1.0;
    molecules_sum += molecules_i * weight_i;
    aerosols_sum += aerosols_i * weight_i;
  }
  molecules = molecules_sum * dx * SOLAR_IRRADIANCE * atmosphere.molecules.scattering;
  aerosols  = aerosols_sum * dx * SOLAR_IRRADIANCE * atmosphere.aerosols.scattering;
}

vec3 PhaseFunction(ScatteringComponent component, float nu) {
  float theta = acos(nu) / PI; // 0<->1
  return texture2D(phase_texture, vec2(theta, component.phaseTextureV)).rgb;
}

// Single-Scattering Texture Pre-Computation -------------------------------------------------------

// The code below is used to store the single scattering (without the phase function applied) in a
// 4D lookup table.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#single_scattering_precomputation

// There is no functional difference to the original code.

vec4 GetScatteringTextureUvwzFromRMuMuSNu(
    float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground) {

  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon.
  float rho = SafeSqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS);
  float u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

  // Discriminant of the quadratic equation for the intersections of the ray (r,mu) with the ground
  // (see RayIntersectsGround).
  float r_mu         = r * mu;
  float discriminant = r_mu * r_mu - r * r + BOTTOM_RADIUS * BOTTOM_RADIUS;
  float u_mu;
  if (ray_r_mu_intersects_ground) {
    // Distance to the ground for the ray (r,mu), and its minimum and maximum values over all mu -
    // obtained for (r,-1) and (r,mu_horizon).
    float d     = -r_mu - SafeSqrt(discriminant);
    float d_min = r - BOTTOM_RADIUS;
    float d_max = rho;
    u_mu        = 0.5 -
           0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min),
                     SCATTERING_TEXTURE_MU_SIZE / 2);
  } else {
    // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum and maximum
    // values over all mu - obtained for (r,1) and (r,mu_horizon).
    float d     = -r_mu + SafeSqrt(discriminant + H * H);
    float d_min = TOP_RADIUS - r;
    float d_max = rho + H;
    u_mu        = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
                           (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
  }

  float d     = DistanceToTopAtmosphereBoundary(BOTTOM_RADIUS, mu_s);
  float d_min = TOP_RADIUS - BOTTOM_RADIUS;
  float d_max = H;
  float a     = (d - d_min) / (d_max - d_min);
  float D     = DistanceToTopAtmosphereBoundary(BOTTOM_RADIUS, MU_S_MIN);
  float A     = (D - d_min) / (d_max - d_min);
  // An ad-hoc function equal to 0 for mu_s = MU_S_MIN (because then d = D and thus a = A), equal to
  // 1 for mu_s = 1 (because then d = d_min and thus a = 0), and with a large slope around mu_s = 0,
  // to get more texture samples near the horizon.
  float u_mu_s =
      GetTextureCoordFromUnitRange(max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

  float u_nu = (nu + 1.0) / 2.0;
  return vec4(u_nu, u_mu_s, u_mu, u_r);
}

void GetRMuMuSNuFromScatteringTextureUvwz(vec4 uvwz, out float r, out float mu, out float mu_s,
    out float nu, out bool ray_r_mu_intersects_ground) {

  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon.
  float rho = H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
  r         = sqrt(rho * rho + BOTTOM_RADIUS * BOTTOM_RADIUS);

  if (uvwz.z < 0.5) {
    // Distance to the ground for the ray (r,mu), and its minimum and maximum values over all mu -
    // obtained for (r,-1) and (r,mu_horizon) - from which we can recover mu:
    float d_min = r - BOTTOM_RADIUS;
    float d_max = rho;
    float d     = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
                                            1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);
    mu = d == 0.0 ? -1.0 : ClampCosine(-(rho * rho + d * d) / (2.0 * r * d));
    ray_r_mu_intersects_ground = true;
  } else {
    // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum and maximum
    // values over all mu - obtained for (r,1) and (r,mu_horizon) - from which we can recover mu:
    float d_min = TOP_RADIUS - r;
    float d_max = rho + H;
    float d     = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
                                            2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);
    mu = d == 0.0 ? 1.0 : ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
    ray_r_mu_intersects_ground = false;
  }

  float x_mu_s = GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
  float d_min  = TOP_RADIUS - BOTTOM_RADIUS;
  float d_max  = H;
  float D      = DistanceToTopAtmosphereBoundary(BOTTOM_RADIUS, MU_S_MIN);
  float A      = (D - d_min) / (d_max - d_min);
  float a      = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
  float d      = d_min + min(a, A) * (d_max - d_min);
  mu_s         = d == 0.0 ? 1.0 : ClampCosine((H * H - d * d) / (2.0 * BOTTOM_RADIUS * d));

  nu = ClampCosine(uvwz.x * 2.0 - 1.0);
}

void GetRMuMuSNuFromScatteringTextureFragCoord(vec3 frag_coord, out float r, out float mu,
    out float mu_s, out float nu, out bool ray_r_mu_intersects_ground) {
  const vec4 SCATTERING_TEXTURE_SIZE = vec4(SCATTERING_TEXTURE_NU_SIZE - 1,
      SCATTERING_TEXTURE_MU_S_SIZE, SCATTERING_TEXTURE_MU_SIZE, SCATTERING_TEXTURE_R_SIZE);
  float      frag_coord_nu           = floor(frag_coord.x / float(SCATTERING_TEXTURE_MU_S_SIZE));
  float      frag_coord_mu_s         = mod(frag_coord.x, float(SCATTERING_TEXTURE_MU_S_SIZE));
  vec4       uvwz =
      vec4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) / SCATTERING_TEXTURE_SIZE;
  GetRMuMuSNuFromScatteringTextureUvwz(uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  // Clamp nu to its valid range of values, given mu and mu_s.
  nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
      mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));
}

void ComputeSingleScatteringTexture(AtmosphereComponents atmosphere,
    sampler2D transmittance_texture, vec3 frag_coord, out vec3 molecules, out vec3 aerosols) {
  float r;
  float mu;
  float mu_s;
  float nu;
  bool  ray_r_mu_intersects_ground;
  GetRMuMuSNuFromScatteringTextureFragCoord(
      frag_coord, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  ComputeSingleScattering(atmosphere, transmittance_texture, r, mu, mu_s, nu,
      ray_r_mu_intersects_ground, molecules, aerosols);
}

// Single-Scattering Texture Lookup ----------------------------------------------------------------

// The code below is used to retrieve the single-scattering values from the lookup tables.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#single_scattering_lookup

// There is no functional difference to the original code.

vec3 GetScattering(sampler3D scattering_texture, float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground) {
  vec4  uvwz = GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  float tex_coord_x = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
  float tex_x       = floor(tex_coord_x);
  float lerp        = tex_coord_x - tex_x;
  vec3  uvw0        = vec3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  vec3  uvw1 = vec3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  return vec3(
      texture(scattering_texture, uvw0) * (1.0 - lerp) + texture(scattering_texture, uvw1) * lerp);
}

vec3 GetScattering(AtmosphereComponents atmosphere, sampler3D single_molecules_scattering_texture,
    sampler3D single_aerosols_scattering_texture, sampler3D multiple_scattering_texture, float r,
    float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground, int scattering_order) {
  if (scattering_order == 1) {
    vec3 molecules = GetScattering(
        single_molecules_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    vec3 aerosols = GetScattering(
        single_aerosols_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    return molecules * PhaseFunction(atmosphere.molecules, nu) +
           aerosols * PhaseFunction(atmosphere.aerosols, nu);
  } else {
    return GetScattering(multiple_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  }
}

// Multiple-Scattering Computation -----------------------------------------------------------------

// The code below is used to compute the amount of light scattered after more than one bounces in
// the atmosphere.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#multiple_scattering

// There is no functional difference to the original code.

vec3 GetIrradiance(sampler2D irradiance_texture, float r, float mu_s);

vec3 ComputeScatteringDensity(AtmosphereComponents atmosphere, sampler2D transmittance_texture,
    sampler3D single_molecules_scattering_texture, sampler3D single_aerosols_scattering_texture,
    sampler3D multiple_scattering_texture, sampler2D irradiance_texture, float r, float mu,
    float mu_s, float nu, int scattering_order) {

  // Compute unit direction vectors for the zenith, the view direction omega and and the sun
  // direction omega_s, such that the cosine of the view-zenith angle is mu, the cosine of the
  // sun-zenith angle is mu_s, and the cosine of the view-sun angle is nu. The goal is to simplify
  // computations below.
  vec3  zenith_direction = vec3(0.0, 0.0, 1.0);
  vec3  omega            = vec3(sqrt(1.0 - mu * mu), 0.0, mu);
  float sun_dir_x        = omega.x == 0.0 ? 0.0 : (nu - mu * mu_s) / omega.x;
  float sun_dir_y        = sqrt(max(1.0 - sun_dir_x * sun_dir_x - mu_s * mu_s, 0.0));
  vec3  omega_s          = vec3(sun_dir_x, sun_dir_y, mu_s);

  const float dphi               = PI / float(SAMPLE_COUNT_SCATTERING_DENSITY);
  const float dtheta             = PI / float(SAMPLE_COUNT_SCATTERING_DENSITY);
  vec3        molecules_aerosols = vec3(0.0);

  // Nested loops for the integral over all the incident directions omega_i.
  for (int l = 0; l < SAMPLE_COUNT_SCATTERING_DENSITY; ++l) {
    float theta                         = (float(l) + 0.5) * dtheta;
    float cos_theta                     = cos(theta);
    float sin_theta                     = sin(theta);
    bool  ray_r_theta_intersects_ground = RayIntersectsGround(r, cos_theta);

    // The distance and transmittance to the ground only depend on theta, so we can compute them in
    // the outer loop for efficiency.
    float distance_to_ground      = 0.0;
    vec3  transmittance_to_ground = vec3(0.0);
    vec3  ground_albedo           = vec3(0.0);
    if (ray_r_theta_intersects_ground) {
      distance_to_ground      = DistanceToBottomAtmosphereBoundary(r, cos_theta);
      transmittance_to_ground = GetTransmittance(transmittance_texture, r, cos_theta,
          distance_to_ground, true /* ray_intersects_ground */);
      ground_albedo           = GROUND_ALBEDO;
    }

    for (int m = 0; m < 2 * SAMPLE_COUNT_SCATTERING_DENSITY; ++m) {
      float phi      = (float(m) + 0.5) * dphi;
      vec3  omega_i  = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
      float domega_i = dtheta * dphi * sin(theta);

      // The radiance L_i arriving from direction omega_i after n-1 bounces is the sum of a term
      // given by the precomputed scattering texture for the (n-1)-th order:
      float nu1               = dot(omega_s, omega_i);
      vec3  incident_radiance = GetScattering(atmosphere, single_molecules_scattering_texture,
          single_aerosols_scattering_texture, multiple_scattering_texture, r, omega_i.z, mu_s, nu1,
          ray_r_theta_intersects_ground, scattering_order - 1);

      // and of the contribution from the light paths with n-1 bounces and whose last bounce is on
      // the ground. This contribution is the product of the transmittance to the ground, the ground
      // albedo, the ground BRDF, and the irradiance received on the ground after n-2 bounces.
      vec3 ground_normal = normalize(zenith_direction * r + omega_i * distance_to_ground);
      vec3 ground_irradiance =
          GetIrradiance(irradiance_texture, BOTTOM_RADIUS, dot(ground_normal, omega_s));
      incident_radiance += transmittance_to_ground * ground_albedo * (1.0 / PI) * ground_irradiance;

      // The radiance finally scattered from direction omega_i towards direction -omega is the
      // product of the incident radiance, the scattering coefficient, and the phase function for
      // directions omega and omega_i (all this summed over all particle types, i.e. Rayleigh and
      // Mie).
      float nu2               = dot(omega, omega_i);
      float molecules_density = GetDensity(atmosphere.molecules.densityTextureV, r - BOTTOM_RADIUS);
      float aerosols_density  = GetDensity(atmosphere.aerosols.densityTextureV, r - BOTTOM_RADIUS);
      molecules_aerosols += incident_radiance *
                            (atmosphere.molecules.scattering * molecules_density *
                                    PhaseFunction(atmosphere.molecules, nu2) +
                                atmosphere.aerosols.scattering * aerosols_density *
                                    PhaseFunction(atmosphere.aerosols, nu2)) *
                            domega_i;
    }
  }
  return molecules_aerosols;
}

vec3 ComputeMultipleScattering(sampler2D transmittance_texture,
    sampler3D scattering_density_texture, float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground) {

  // The integration step, i.e. the length of each integration interval.
  float dx = DistanceToNearestAtmosphereBoundary(r, mu, ray_r_mu_intersects_ground) /
             float(SAMPLE_COUNT_MULTI_SCATTERING);
  // Integration loop.
  vec3 molecules_aerosols_sum = vec3(0.0);
  for (int i = 0; i <= SAMPLE_COUNT_MULTI_SCATTERING; ++i) {
    float d_i = float(i) * dx;

    // The r, mu and mu_s parameters at the current integration point (see the single scattering
    // section for a detailed explanation).
    float r_i    = ClampRadius(sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r));
    float mu_i   = ClampCosine((r * mu + d_i) / r_i);
    float mu_s_i = ClampCosine((r * mu_s + d_i * nu) / r_i);

    // The Rayleigh and Mie multiple scattering at the current sample point.
    vec3 molecules_aerosols_i =
        GetScattering(
            scattering_density_texture, r_i, mu_i, mu_s_i, nu, ray_r_mu_intersects_ground) *
        GetTransmittance(transmittance_texture, r, mu, d_i, ray_r_mu_intersects_ground) * dx;
    // Sample weight (from the trapezoidal rule).
    float weight_i = (i == 0 || i == SAMPLE_COUNT_MULTI_SCATTERING) ? 0.5 : 1.0;
    molecules_aerosols_sum += molecules_aerosols_i * weight_i;
  }
  return molecules_aerosols_sum;
}

// Multiple-Scattering Texture Pre-Computation -----------------------------------------------------

// The code below is used to store the multiple scattering (with the phase function applied) in a
// 4D lookup table.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#multiple_scattering_precomputation

// There is no functional difference to the original code.

vec3 ComputeScatteringDensityTexture(AtmosphereComponents atmosphere,
    sampler2D transmittance_texture, sampler3D single_molecules_scattering_texture,
    sampler3D single_aerosols_scattering_texture, sampler3D multiple_scattering_texture,
    sampler2D irradiance_texture, vec3 frag_coord, int scattering_order) {
  float r;
  float mu;
  float mu_s;
  float nu;
  bool  ray_r_mu_intersects_ground;
  GetRMuMuSNuFromScatteringTextureFragCoord(
      frag_coord, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  return ComputeScatteringDensity(atmosphere, transmittance_texture,
      single_molecules_scattering_texture, single_aerosols_scattering_texture,
      multiple_scattering_texture, irradiance_texture, r, mu, mu_s, nu, scattering_order);
}

vec3 ComputeMultipleScatteringTexture(sampler2D transmittance_texture,
    sampler3D scattering_density_texture, vec3 frag_coord, out float nu) {
  float r;
  float mu;
  float mu_s;
  bool  ray_r_mu_intersects_ground;
  GetRMuMuSNuFromScatteringTextureFragCoord(
      frag_coord, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  return ComputeMultipleScattering(transmittance_texture, scattering_density_texture, r, mu, mu_s,
      nu, ray_r_mu_intersects_ground);
}

// Compute Irradiance ------------------------------------------------------------------------------

// The code below is used to compute the irradiance received from the Sun and from the sky at a
// given altitude.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#irradiance

// There is no functional difference to the original code.

vec3 ComputeDirectIrradiance(sampler2D transmittance_texture, float r, float mu_s) {

  float alpha_s = SUN_ANGULAR_RADIUS;
  // Approximate average of the cosine factor mu_s over the visible fraction of
  // the Sun disc.
  float average_cosine_factor =
      mu_s < -alpha_s
          ? 0.0
          : (mu_s > alpha_s ? mu_s : (mu_s + alpha_s) * (mu_s + alpha_s) / (4.0 * alpha_s));

  return SOLAR_IRRADIANCE *
         GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu_s) *
         average_cosine_factor;
}

vec3 ComputeIndirectIrradiance(AtmosphereComponents atmosphere,
    sampler3D single_molecules_scattering_texture, sampler3D single_aerosols_scattering_texture,
    sampler3D multiple_scattering_texture, float r, float mu_s, int scattering_order) {

  const float dphi   = PI / float(SAMPLE_COUNT_INDIRECT_IRRADIANCE);
  const float dtheta = PI / float(SAMPLE_COUNT_INDIRECT_IRRADIANCE);

  vec3 result  = vec3(0.0);
  vec3 omega_s = vec3(sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);
  for (int j = 0; j < SAMPLE_COUNT_INDIRECT_IRRADIANCE / 2; ++j) {
    float theta = (float(j) + 0.5) * dtheta;
    for (int i = 0; i < 2 * SAMPLE_COUNT_INDIRECT_IRRADIANCE; ++i) {
      float phi    = (float(i) + 0.5) * dphi;
      vec3  omega  = vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
      float domega = dtheta * dphi * sin(theta);

      float nu = dot(omega, omega_s);
      result += GetScattering(atmosphere, single_molecules_scattering_texture,
                    single_aerosols_scattering_texture, multiple_scattering_texture, r, omega.z,
                    mu_s, nu, false /* ray_r_theta_intersects_ground */, scattering_order) *
                omega.z * domega;
    }
  }
  return result;
}

// Irradiance-Texture Pre-Computation --------------------------------------------------------------

// The code below is used to store the direct and indirect irradiance received at any altitude in 2D
// lookup tables.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#irradiance_precomputation

// There is no functional difference to the original code.

vec2 GetIrradianceTextureUvFromRMuS(float r, float mu_s) {
  float x_r    = (r - BOTTOM_RADIUS) / (TOP_RADIUS - BOTTOM_RADIUS);
  float x_mu_s = mu_s * 0.5 + 0.5;
  return vec2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
      GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
}

void GetRMuSFromIrradianceTextureUv(vec2 uv, out float r, out float mu_s) {
  float x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
  float x_r    = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
  r            = BOTTOM_RADIUS + x_r * (TOP_RADIUS - BOTTOM_RADIUS);
  mu_s         = ClampCosine(2.0 * x_mu_s - 1.0);
}

vec3 ComputeDirectIrradianceTexture(sampler2D transmittance_texture, vec2 frag_coord) {
  float r;
  float mu_s;
  GetRMuSFromIrradianceTextureUv(
      frag_coord / vec2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT), r, mu_s);
  return ComputeDirectIrradiance(transmittance_texture, r, mu_s);
}

vec3 ComputeIndirectIrradianceTexture(AtmosphereComponents atmosphere,
    sampler3D single_molecules_scattering_texture, sampler3D single_aerosols_scattering_texture,
    sampler3D multiple_scattering_texture, vec2 frag_coord, int scattering_order) {
  float r;
  float mu_s;
  GetRMuSFromIrradianceTextureUv(
      frag_coord / vec2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT), r, mu_s);
  return ComputeIndirectIrradiance(atmosphere, single_molecules_scattering_texture,
      single_aerosols_scattering_texture, multiple_scattering_texture, r, mu_s, scattering_order);
}

// Irradiance-Texture Lookup -----------------------------------------------------------------------

// The code below is used to retrieve the Sun and sky irradiance values from any altitude from the
// lookup tables.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#irradiance_lookup

// There is no functional difference to the original code.

vec3 GetIrradiance(sampler2D irradiance_texture, float r, float mu_s) {
  vec2 uv = GetIrradianceTextureUvFromRMuS(r, mu_s);
  return vec3(texture(irradiance_texture, uv));
}

// Combining Single- and Multiple-Scattering Contributions -----------------------------------------

// The code below is used to retrieve single molecule-scattrering + multiple-scattering, and single
// aerosol-scattering contributions from the 4D textures.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#rendering

// The only difference is that we removed the optimized case where (monochrome) aerosol scattering
// is stored in the alpha channel of the scattering texture.

void GetCombinedScattering(sampler3D multiple_scattering_texture,
    sampler3D single_aerosols_scattering_texture, float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground, out vec3 multiple_scattering,
    out vec3 single_aerosols_scattering) {
  vec4  uvwz = GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  float tex_coord_x = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
  float tex_x       = floor(tex_coord_x);
  float lerp        = tex_coord_x - tex_x;
  vec3  uvw0        = vec3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  vec3  uvw1 = vec3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);

  multiple_scattering = vec3(texture(multiple_scattering_texture, uvw0) * (1.0 - lerp) +
                             texture(multiple_scattering_texture, uvw1) * lerp);
  single_aerosols_scattering =
      vec3(texture(single_aerosols_scattering_texture, uvw0) * (1.0 - lerp) +
           texture(single_aerosols_scattering_texture, uvw1) * lerp);
}

// Commpute Sky Radiance ---------------------------------------------------------------------------

// The code below is used to retrieve the color of the sky from the pre-computed textures.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#rendering_sky

// The only difference is that we removed the code for light shafts, as this is currently not
// supported by CosmoScout VR.

vec3 GetSkyRadiance(AtmosphereComponents atmosphere, sampler2D transmittance_texture,
    sampler3D multiple_scattering_texture, sampler3D single_aerosols_scattering_texture,
    vec3 camera, vec3 view_ray, vec3 sun_direction, out vec3 transmittance) {
  // Compute the distance to the top atmosphere boundary along the view ray, assuming the viewer is
  // in space (or NaN if the view ray does not intersect the atmosphere).
  float r   = length(camera);
  float rmu = dot(camera, view_ray);
  float distance_to_top_atmosphere_boundary =
      -rmu - sqrt(rmu * rmu - r * r + TOP_RADIUS * TOP_RADIUS);
  // If the viewer is in space and the view ray intersects the atmosphere, move the viewer to the
  // top atmosphere boundary (along the view ray):
  if (distance_to_top_atmosphere_boundary > 0.0) {
    camera = camera + view_ray * distance_to_top_atmosphere_boundary;
    r      = TOP_RADIUS;
    rmu += distance_to_top_atmosphere_boundary;
  } else if (r > TOP_RADIUS) {
    // If the view ray does not intersect the atmosphere, simply return 0.
    transmittance = vec3(1.0);
    return vec3(0.0);
  }
  // Compute the r, mu, mu_s and nu parameters needed for the texture lookups.
  float mu                         = rmu / r;
  float mu_s                       = dot(camera, sun_direction) / r;
  float nu                         = dot(view_ray, sun_direction);
  bool  ray_r_mu_intersects_ground = RayIntersectsGround(r, mu);

  transmittance = ray_r_mu_intersects_ground
                      ? vec3(0.0)
                      : GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu);

  vec3 multiple_scattering;
  vec3 single_aerosols_scattering;
  GetCombinedScattering(multiple_scattering_texture, single_aerosols_scattering_texture, r, mu,
      mu_s, nu, ray_r_mu_intersects_ground, multiple_scattering, single_aerosols_scattering);

  return multiple_scattering * PhaseFunction(atmosphere.molecules, nu) +
         single_aerosols_scattering * PhaseFunction(atmosphere.aerosols, nu);
}

// Arial Perspective -------------------------------------------------------------------------------

// The code below is used to retrieve the amount of inscattered light between two points in the
// atmosphere.

// An explanation of the following methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#rendering_aerial_perspective

// The only difference is that we removed the code for light shafts, as this is currently not
// supported by CosmoScout VR.

vec3 GetSkyRadianceToPoint(AtmosphereComponents atmosphere, sampler2D transmittance_texture,
    sampler3D multiple_scattering_texture, sampler3D single_aerosols_scattering_texture,
    vec3 camera, vec3 point, vec3 sun_direction, out vec3 transmittance) {
  // Compute the distance to the top atmosphere boundary along the view ray, assuming the viewer is
  // in space (or NaN if the view ray does not intersect the atmosphere).
  vec3  view_ray = normalize(point - camera);
  float r        = length(camera);
  float rmu      = dot(camera, view_ray);
  float distance_to_top_atmosphere_boundary =
      -rmu - sqrt(rmu * rmu - r * r + TOP_RADIUS * TOP_RADIUS);
  // If the viewer is in space and the view ray intersects the atmosphere, move the viewer to the
  // top atmosphere boundary (along the view ray):
  if (distance_to_top_atmosphere_boundary > 0.0) {
    camera = camera + view_ray * distance_to_top_atmosphere_boundary;
    r      = TOP_RADIUS;
    rmu += distance_to_top_atmosphere_boundary;
  }

  // Compute the r, mu, mu_s and nu parameters for the first texture lookup.
  float mu                         = rmu / r;
  float mu_s                       = dot(camera, sun_direction) / r;
  float nu                         = dot(view_ray, sun_direction);
  float d                          = length(point - camera);
  bool  ray_r_mu_intersects_ground = RayIntersectsGround(r, mu);

  transmittance = GetTransmittance(transmittance_texture, r, mu, d, ray_r_mu_intersects_ground);

  vec3 multiple_scattering;
  vec3 single_aerosols_scattering;

  GetCombinedScattering(multiple_scattering_texture, single_aerosols_scattering_texture, r, mu,
      mu_s, nu, ray_r_mu_intersects_ground, multiple_scattering, single_aerosols_scattering);

  // Compute the r, mu, mu_s and nu parameters for the second texture lookup.
  float r_p    = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  float mu_p   = (r * mu + d) / r_p;
  float mu_s_p = (r * mu_s + d * nu) / r_p;

  vec3 multiple_scattering_p;
  vec3 single_aerosols_scattering_p;

  GetCombinedScattering(multiple_scattering_texture, single_aerosols_scattering_texture, r_p, mu_p,
      mu_s_p, nu, ray_r_mu_intersects_ground, multiple_scattering_p, single_aerosols_scattering_p);

  // Combine the lookup results to get the scattering between camera and point.
  multiple_scattering = multiple_scattering - transmittance * multiple_scattering_p;
  single_aerosols_scattering =
      single_aerosols_scattering - transmittance * single_aerosols_scattering_p;

  // Hack to avoid rendering artifacts when the sun is below the horizon.
  single_aerosols_scattering = single_aerosols_scattering * smoothstep(0.0, 0.01, mu_s);

  return multiple_scattering * PhaseFunction(atmosphere.molecules, nu) +
         single_aerosols_scattering * PhaseFunction(atmosphere.aerosols, nu);
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

vec3 GetSunAndSkyIrradiance(sampler2D transmittance_texture, sampler2D irradiance_texture,
    vec3 point, vec3 sun_direction, out vec3 sky_irradiance) {
  float r    = length(point);
  float mu_s = dot(point, sun_direction) / r;

  // Indirect irradiance.
  sky_irradiance = GetIrradiance(irradiance_texture, r, mu_s);

  // Direct irradiance.
  return SOLAR_IRRADIANCE * GetTransmittanceToSun(transmittance_texture, r, mu_s);
}
