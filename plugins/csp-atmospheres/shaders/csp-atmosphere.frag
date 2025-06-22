////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2014 David Hoskins
// SPDX-FileCopyrightText: 2013 Nikita Miropolskiy
// SPDX-License-Identifier: MIT

#version 430

// inputs
in VaryingStruct {
  vec3 rayDir;
  vec3 rayOrigin;
  vec2 texcoords;
}
vsIn;

// constants
const float PI = 3.141592653589793;

// uniforms
#if HDR_SAMPLES > 0
uniform sampler2DMS uColorBuffer;
uniform sampler2DMS uDepthBuffer;
#else
uniform sampler2D uColorBuffer;
uniform sampler2D uDepthBuffer;
#endif

uniform vec3      uSunDir;
uniform vec3      uSunInfo; // x: sun luminance, y: sun illuminance, z: sun angular radius
uniform float     uTime;
uniform mat4      uMatM;
uniform mat4      uMatMVP;
uniform mat4      uMatScale;
uniform mat4      uMatInvP;
uniform float     uWaterLevel;
uniform sampler2D uCloudTexture;
uniform float     uCloudAltitude;
uniform sampler3D uLimbLuminanceTexture;
uniform vec3      uShadowCoordinates;
uniform sampler3D uNoiseTexture;
uniform sampler2D uNoiseTexture2D;
uniform sampler2D uCloudTypeTexture;
uniform float     uTestUniform;

// outputs
layout(location = 0) out vec3 oColor;

// -------------------------------------------------------------------------------------------------

// Each atmospheric model will implement these three methods. We forward-declare them here. The
// actual implementation comes from the model's shader which is linked to this shader.

// This will return true or false depending on whether the atmosphere model supports refraction.
bool RefractionSupported();

// This will return the view ray after refraction by the atmosphere after it travelled all the way
// to the end of the atmosphere.
vec3 GetRefractedRay(vec3 camera, vec3 ray, out bool hitsGround);

// Returns the sky luminance (in cd/m^2) along the segment from 'camera' to the nearest
// atmosphere boundary in direction 'viewRay', as well as the transmittance along this segment.
vec3 GetSkyLuminance(vec3 camera, vec3 viewRay, vec3 sunDirection, out vec3 transmittance);

// Returns the sky luminance (in cd/m^2) along the segment from 'camera' to 'p', as well as the
// transmittance along this segment.
vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 p, vec3 sunDirection, out vec3 transmittance);

// Returns the sun and sky illuminance (in lux) received on a surface patch located at 'p'.
vec3 GetSunAndSkyIlluminance(vec3 p, vec3 sunDirection, out vec3 skyIlluminance);

// -------------------------------------------------------------------------------------------------

// This will be replaced by the eclipse shader code.
// ECLIPSE_SHADER_SNIPPET

// -------------------------------------------------------------------------------------------------

// These noise algorithms are based on implementations by various authors from
// shadertoy.com, which are all available under the MIT License. See the respective links
// in the comments below.

// Hash function
// MIT License, https://www.shadertoy.com/view/4djSRW
// Copyright (c) 2014 David Hoskins.

// 3 out, 3 in...
vec3 hash33(vec3 p3) {
  p3 = fract(p3 * vec3(.1031, .1030, .0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}

// 3D Simplex Noise
// MIT License, https://www.shadertoy.com/view/XsX3zB
// Copyright (c) 2013 Nikita Miropolskiy
float simplex3D(vec3 p) {

  // skew constants for 3D simplex functions
  const float F3 = 0.3333333;
  const float G3 = 0.1666667;

  // 1. find current tetrahedron T and it's four vertices
  // s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices
  // x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertice

  // calculate s and x
  vec3 s = floor(p + dot(p, vec3(F3)));
  vec3 x = p - s + dot(s, vec3(G3));

  // calculate i1 and i2
  vec3 e  = step(vec3(0.0), x - x.yzx);
  vec3 i1 = e * (1.0 - e.zxy);
  vec3 i2 = 1.0 - e.zxy * (1.0 - e);

  // x1, x2, x3
  vec3 x1 = x - i1 + G3;
  vec3 x2 = x - i2 + 2.0 * G3;
  vec3 x3 = x - 1.0 + 3.0 * G3;

  // 2. find four surflets and store them in d
  vec4 w, d;

  // calculate surflet weights
  w.x = dot(x, x);
  w.y = dot(x1, x1);
  w.z = dot(x2, x2);
  w.w = dot(x3, x3);

  // w fades from 0.6 at the center of the surflet to 0.0 at the margin
  w = max(0.6 - w, 0.0);

  // calculate surflet components
  d.x = dot(-0.5 + hash33(s), x);
  d.y = dot(-0.5 + hash33(s + i1), x1);
  d.z = dot(-0.5 + hash33(s + i2), x2);
  d.w = dot(-0.5 + hash33(s + 1.0), x3);

  // multiply d by w^4
  w *= w;
  w *= w;
  d *= w;

  // 3. return the sum of the four surflets
  return dot(d, vec4(52.0)) * 0.5 + 0.5;
}

// Directional artifacts can be reduced by rotating each octave
float simplex3DFractal(vec3 m) {

  // const matrices for 3D rotation
  const mat3 rot1 = mat3(-0.37, 0.36, 0.85, -0.14, -0.93, 0.34, 0.92, 0.01, 0.4);
  const mat3 rot2 = mat3(-0.55, -0.39, 0.74, 0.33, -0.91, -0.24, 0.77, 0.12, 0.63);
  const mat3 rot3 = mat3(-0.71, 0.52, -0.47, -0.08, -0.72, -0.68, -0.7, -0.45, 0.56);

  return 0.5333333 * simplex3D(m * rot1) + 0.2666667 * simplex3D(2.0 * m * rot2) +
         0.1333333 * simplex3D(4.0 * m * rot3) + 0.0666667 * simplex3D(8.0 * m);
}

// -------------------------------------------------------------------------------------------------

// Tonemapping code and color space conversions.
// http://filmicworlds.com/blog/filmic-tonemapping-operators/
float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 uncharted2Tonemap(vec3 c) {
  return ((c * (A * c + C * B) + D * E) / (c * (A * c + B) + D * F)) - E / F;
}

vec3 tonemap(vec3 c) {
  c               = uncharted2Tonemap(10.0 * c);
  vec3 whiteScale = vec3(1.0) / uncharted2Tonemap(vec3(W));
  return c * whiteScale;
}

float linearToSRGB(float c) {
  if (c <= 0.0031308)
    return 12.92 * c;
  else
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

vec3 linearToSRGB(vec3 c) {
  return vec3(linearToSRGB(c.r), linearToSRGB(c.g), linearToSRGB(c.b));
}

vec3 sRGBtoLinear(vec3 c) {
  vec3 bLess = step(vec3(0.04045), c);
  return mix(c / vec3(12.92), pow((c + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
}

float sRGBtoLinear(float c) {
  float bLess = step(0.04045, c);
  return mix(c / 12.92, pow((c + 0.055) / 1.055, 2.4), bLess);
}

// -------------------------------------------------------------------------------------------------

// Compute intersections of a ray with a sphere. Two T parameters are returned -- if no intersection
// is found, the first will be larger than the second. The T parameters can be nagative. In this
// case, the intersections are behind the origin (in negative ray direction).
vec2 intersectSphere(vec3 rayOrigin, vec3 rayDir, float radius) {
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
vec2 intersectAtmosphere(vec3 rayOrigin, vec3 rayDir) {
  return intersectSphere(rayOrigin, rayDir, ATMOSPHERE_RADIUS);
}

// Computes the intersections of a ray with the planet.
vec2 intersectPlanetsphere(vec3 rayOrigin, vec3 rayDir) {
  return intersectSphere(rayOrigin, rayDir, PLANET_RADIUS);
}

// Computes the intersections of a ray with the ocean.
vec2 intersectOceansphere(vec3 rayOrigin, vec3 rayDir) {
  return intersectSphere(rayOrigin, rayDir, uWaterLevel + PLANET_RADIUS);
}

// -------------------------------------------------------------------------------------------------

// Computes the longitude and latitude for any cartesian position relative to the unit sphere
// centered at the origin.
vec2 getLngLat(vec3 position) {
  vec2 result;
  result.x = atan(position.x, position.z);
  result.y = asin(position.y / length(position));
  return result;
}

// -------------------------------------------------------------------------------------------------

// Returns the background color at the current pixel. If multisampling is used, we take the average
// color.
vec3 getFramebufferColor(vec2 texcoords) {
#if HDR_SAMPLES > 0
  vec3 color = vec3(0.0);
  for (int i = 0; i < HDR_SAMPLES; ++i) {
    color += texelFetch(uColorBuffer, ivec2(texcoords * textureSize(uColorBuffer)), i).rgb;
  }
  return color / HDR_SAMPLES;
#else
  return texture(uColorBuffer, texcoords).rgb;
#endif
}

// Returns the depth at the current pixel. If multisampling is used, we take the minimum depth.
float getFramebufferDepth(vec2 texcoords) {
#if HDR_SAMPLES > 0
  float depth = 1.0;
  for (int i = 0; i < HDR_SAMPLES; ++i) {
    depth = min(depth, texelFetch(uDepthBuffer, ivec2(texcoords * textureSize(uDepthBuffer)), i).r);
  }
  return depth;
#else
  return texture(uDepthBuffer, texcoords).r;
#endif
}

// Using acos is not very stable for small angles. This function is used to compute the angle
// between two vectors in a more stable way.
float angleBetweenVectors(vec3 u, vec3 v) {
  return 2.0 * asin(0.5 * length(u - v));
}

// This methods returns a color from the framebuffer which most likely represents what an observer
// would see if looking in the direction of the given ray. If the ray hits the ground, black is
// returned. If the ray is refracted around the planet, we cannot sample the framebuffer but return
// black as well. However, if the ray would hit the Sun, the color of the Sun is returned.
vec3 getRefractedFramebufferColor(vec3 rayOrigin, vec3 rayDir, out vec3 refractedRay) {

  // First, we assume that the refracted ray will leave the atmosphere unblocked. We compute the
  // texture coordinates where the ray would hit the framebuffer.
  bool hitsGround;
  refractedRay = GetRefractedRay(rayOrigin, rayDir, hitsGround);

  if (hitsGround) {
    return vec3(0, 0, 0);
  }

  vec4 texcoords = uMatMVP * vec4(refractedRay, 0.0);
  texcoords.xy   = texcoords.xy / texcoords.w * 0.5 + 0.5;

  // We can only sample the color buffer if the point is inside the screen.
  bool inside = all(lessThan(texcoords.xy, vec2(1.0))) && all(greaterThan(texcoords.xy, vec2(0.0)));

  // Also, we check the depth buffer to see if the point is occluded. If it is, we do not sample
  // the color buffer.
  bool occluded = getFramebufferDepth(texcoords.xy) > 0.0;
  
  if (inside && !occluded) {
    return getFramebufferColor(texcoords.xy);
  }

  float sunAngularRadius = uSunInfo.z;
  float sunColor         = 0.0;

  if (angleBetweenVectors(refractedRay, uSunDir) < sunAngularRadius) {
    sunColor = uSunInfo.x;
  }

  return vec3(sunColor);
}

// Returns the distance to the surface of the depth buffer at the current pixel. If the depth of the
// next opaque object is very close to the far end of our depth buffer, we will get jittering
// artifacts. That's the case if we are next to a satellite or on a moon and look towards a planet
// with an atmosphere. In this case, start and end of the ray through the atmosphere basically map
// to the same depth. Therefore, if the depth is really far away (close to zero) we compute the
// intersection with the planet analytically and blend to this value instead. This means, if you are
// close to a satellite, mountains of the planet below cannot poke through the atmosphere anymore.
float getSurfaceDistance(vec3 rayOrigin, vec3 rayDir) {
  float depth = getFramebufferDepth(vsIn.texcoords);

  // If the fragment is really far away, the inverse reverse infinite projection divides by zero.
  // So we add a minimum threshold here.
  depth = max(depth, 0.0000001);

  // We compute the observer-centric distance to the current pixel. uMatScale is required to apply
  // the non-uniform scale of the ellipsoidal atmosphere to the reconstructed position.
  vec4 fragDir = uMatInvP * vec4(2.0 * vsIn.texcoords - 1, 2 * depth - 1, 1);
  fragDir /= fragDir.w;
  fragDir       = uMatScale * fragDir;
  float depthMS = length(fragDir.xyz);

  // Fade to an analytical sphere on the far end of the depth buffer.
  const float START_DEPTH_FADE = 0.001;
  const float END_DEPTH_FADE   = 0.00001;

  // We are only using the depth approximation if depth is smaller than START_DEPTH_FADE and if
  // the observer is outside of the atmosphere.
  if (depth < START_DEPTH_FADE && length(rayOrigin) > ATMOSPHERE_RADIUS) {
    vec2  planetIntersections     = intersectPlanetsphere(rayOrigin, rayDir);
    vec2  atmosphereIntersections = intersectAtmosphere(rayOrigin, rayDir);
    float simpleDepthMS =
        planetIntersections.y > 0.0 ? planetIntersections.x : atmosphereIntersections.y;
    return mix(simpleDepthMS, depthMS,
        clamp((depth - END_DEPTH_FADE) / (START_DEPTH_FADE - END_DEPTH_FADE), 0.0, 1.0));
  }

  return depthMS;
}

// -------------------------------------------------------------------------------------------------

// Returns a hard-coded color scale for a given ocean depth. Could be configurable in future.
vec4 getOceanShade(float d) {
  const float steps[5]  = float[](0.0, 50.0, 100.0, 500.0, 2000.0);
  const vec4  colors[5] = vec4[](vec4(0.8, 0.8, 1, 0.0), vec4(0.3, 0.4, 0.6, 0.3),
      vec4(0.2, 0.3, 0.4, 0.4), vec4(0.1, 0.2, 0.3, 0.8), vec4(0.03, 0.05, 0.1, 1.0));
  for (int i = 0; i < 4; ++i) {
    if (d <= steps[i + 1])
      return mix(colors[i], colors[i + 1], vec4(d - steps[i]) / (steps[i + 1] - steps[i]));
  }
  return colors[4];
}

// -------------------------------------------------------------------------------------------------

float remap(float v, float min_old, float max_old, float min_new, float max_new){
  float v_in_0_1 = (v - min_old) / (max_old - min_old);
  return clamp(v_in_0_1 * (max_new - min_new) + min_new, min(min_new, max_new), max(max_new, min_new));
}

float INFINITY = 1 / 0.;

// heights between which clouds appear
float CUMULONIMBUS_START_HEIGHT = 1500;
float CUMULONIMBUS_END_HEIGHT = 5000;
float COVERAGE_MULTIPLIER = 2;
float CLOUD_BASE_FRACTION = 0.;

// cloud types are remapped from [0,1] so that all values above this become 1 
float CLOUD_COVER_MAX = .8;

// fraction of the cloud layer thickness by which the thickness is locally varying at high frequency
float CLOUD_HEIGHT_VARIATION = .1;

// high frequency noises begin to fade at this distance
float HF_FADE_DISTANCE = 10000;
// high frequency noises have faded to .5 at this distance
float HF_END_DISTANCE = 100000;

// low frequency noises begin to fade at this distance
float LF_FADE_DISTANCE = 500000;
// low frequency noises have faded to .5 at this distance
float LF_END_DISTANCE = 2000000;

// parameter for converting cloud density in [0, 1] to density along path in 1/meter
// Source: I made it up
float DENSITY_MULTIPLIER = 2e-3;
// how much light gets absorbed relative to being scattered
float ABSORBED_FRACTION = 0.;

// parameter for tuning the intensity of the (not physically based) multiscattering approximation
float MS_INTENSITY = .2;

// parameter for setting the clouds to a fancy color
vec3 CLOUD_COLOR = vec3(1.);

// longer segments are cut to this length in scattering calculations to not get erroneously high
// amounts of inscattering over very long segments
float MAXIMUM_DIST_BETWEEN_SAMPLES = 250;

// adaptive step size parameters
float CLOSE_STEP = 50;
float MID_STEP = 100;
float FAR_STEP = 200;
float MID_DISTANCE = 70000;
float FAR_DISTANCE = 200000;
int MAXIMUM_SAMPLES = 800;
// get the cloud type at these texture coordinates
// adds high frequency noises to the values from the cloud texture to replace coarse
// bilinear interpolation artifacts with smaller artifacts that are harder to notice
// returns the noises alongside the result
vec4 GetLocalCloudType(vec2 texCoords){
  float density = remap(textureLod(uCloudTexture, texCoords, 0).r, 0, CLOUD_COVER_MAX, 0, 1);
  vec4 noiseSample = textureLod(uNoiseTexture2D, texCoords * 100, 0);
  // map noises from [0, 1] to [-.5, .5]
  float worleyNoise = noiseSample.b - .5;
  float perlinNoise = noiseSample.r - .5;
  // only the fringes of the clouds should be broken up, the cores should not become noisy
  float noiseWeight = remap(density, .7, 0, 0, 1);
  float allWeight = remap(density, .2, .1, 1, 0);
  float cloudType = clamp(allWeight * remap(density + noiseWeight * (worleyNoise + perlinNoise), 0., 1., -.5, 1.), 0., 0.99);
  return vec4(cloudType, noiseSample);
}

// get the low-fidelity cloud prior like in the Nubis cloud system
// see the 2022 SIGGRAPH talk by Andrew Schneider for what the vertical profile is for
// this implementation differs from Nubis by taking only one texture for cloud type instead of combining a top and bottom type
// returns cloud prior, low and high frequency modes
vec3 GetVerticalProfile(vec3 position){
  vec2 lngLat = getLngLat(position);
  vec2 texCoords = vec2(lngLat.x / (2 * PI) + 0.5, 1.0 - lngLat.y / PI + 0.5);
  vec4 hcomp_with_noise = GetLocalCloudType(texCoords);
  float cloudType = hcomp_with_noise.r;
  vec3 noiseSample = hcomp_with_noise.gba;
  float endHeight = CUMULONIMBUS_END_HEIGHT * (1 - CLOUD_HEIGHT_VARIATION * noiseSample.g);
  float topAltitude = PLANET_RADIUS + endHeight;
  float thickness = endHeight - CUMULONIMBUS_START_HEIGHT;
  // "progress" in cloud from bottom to top in range 0 to 1
  float height_in_cloud = remap(length(position), PLANET_RADIUS + CUMULONIMBUS_START_HEIGHT, topAltitude, 0, 1);
  vec4 cloudConfig = textureLod(uCloudTypeTexture, vec2(cloudType, 1-height_in_cloud), 0);
  return vec3(cloudType > 0 ? pow(cloudConfig.r, 1.5) : 0., cloudConfig.g, cloudConfig.b);
}

// get the density of clouds at a position in 3d space
float getCumuloNimbusDensity(vec3 position, vec3 cam_pos, bool high_res = true){
  vec3 cloudConfig = GetVerticalProfile(position);
  float cloudBase = cloudConfig.r;
  float erosionStrength = cloudConfig.g;
  float hfStrength = cloudConfig.b;
  float cameraDist = length(cam_pos - position);
  vec4 noise2D = textureLod(uNoiseTexture2D, getLngLat(position) * 5, 0);

  float cloudDensity = remap(pow(1 - exp(-COVERAGE_MULTIPLIER * cloudBase), .4), .1, 1, 0, 1);
  
  float lfInfluence = remap(erosionStrength, 0, .8, .05, 1.4);
  float hfInfluence = .1 + .6 * hfStrength;
  if(cameraDist < LF_END_DISTANCE){
    vec4 lfNoises = textureLod(uNoiseTexture, position / 15000, 0);
    // blend between worley and perlin noises using a noise at a different frequency to reduce repetition
    float blended_lf_noise = mix(lfNoises.r, 1 - lfNoises.b, noise2D.r);
    // when camDist is in the fade out range, the noise is mixed with 0.5
    blended_lf_noise = mix(blended_lf_noise, .5, remap(cameraDist, LF_FADE_DISTANCE, LF_END_DISTANCE, 0, 1));
    // using the formula from Andrew Schneider's SIGGRAPH presentations on Nubis
    cloudDensity = clamp(lfInfluence * blended_lf_noise - (lfInfluence - cloudDensity), 0, 1);
    
    if(high_res && cameraDist < HF_END_DISTANCE){
      vec4 hf_noises = textureLod(uNoiseTexture, position / 3521, 0);
      float blended_hf_noise = mix(hf_noises.r, 1 - hf_noises.b, noise2D.b);
      blended_hf_noise = mix(blended_hf_noise, .5,  remap(cameraDist, HF_FADE_DISTANCE, HF_END_DISTANCE, 0, 1));
      cloudDensity = clamp(hfInfluence * blended_hf_noise - (hfInfluence - cloudDensity), 0, 1);
    }else{
      cloudDensity = clamp(hfInfluence * .5 - (hfInfluence - cloudDensity), 0, 1);
    }
  }else{
    // reduce density by assuming noise=0.5 
    // without this operation, the cloud would become more dense at the LOD region border
    // MUST BE PERFORMED FOR ALL FUTURE NOISE SCALES TO AVOID DISCONTINUITIES 
    cloudDensity = clamp(lfInfluence * .5 - (lfInfluence - cloudDensity), 0, 1);
    cloudDensity = clamp(hfInfluence * .5 - (hfInfluence - cloudDensity), 0, 1);
  }
  if(isnan(cloudDensity)){
    cloudDensity = 0;
  }
  return remap(cloudDensity, .1, 1, 0, 1);
}

// cheap check if a location is guaranteed to not contain any clouds
// very similar to GetVerticalProfile but avoids the second texture look-up
bool CumuloNimbusGuaranteedFree(vec3 position){
  vec2 lngLat = getLngLat(position);
  vec2 texCoords = vec2(lngLat.x / (2 * PI) + 0.5, 1.0 - lngLat.y / PI + 0.5);
  vec4 hcomp_with_noise = GetLocalCloudType(texCoords);
  float cloudType = hcomp_with_noise.r;
  vec3 noiseSample = hcomp_with_noise.gba;
  float endHeight = CUMULONIMBUS_END_HEIGHT * (1 - CLOUD_HEIGHT_VARIATION * noiseSample.g);//localMaxHeight(position);
  float topAltitude = PLANET_RADIUS + endHeight;
  float thickness = endHeight - CUMULONIMBUS_START_HEIGHT;
  // "progress" in cloud from bottom to top in range 0 to 1
  float height_in_cloud = remap(length(position), PLANET_RADIUS + CUMULONIMBUS_START_HEIGHT, topAltitude, 0, 1);
  // when the position is above this function of the cloud type, it is assumed cloud-free
  // you might have to change this when modifying the cloud type texture
  return height_in_cloud > pow(cloudType, .2) * 1.2;
}

// calculate a cheap approximation of the maximum distance that can be safely skipped
float CumuloNimbusFreeDistance(vec3 position, vec3 dir, float dist_from_camera, float interval_end){
  if(!CumuloNimbusGuaranteedFree(position)){
    return 0;
  }else{
    int num_samples = 10;
    float dist_to_cover = interval_end - dist_from_camera;
    // further away from the camera, slight flickering of small clouds matters less
    float base_step = remap(dist_from_camera, 0, 200000, 100, 2000);
    if(num_samples * base_step > dist_to_cover){
      base_step = dist_to_cover / num_samples;
    }
    float distance = base_step;
    float last_distance = 0;
    bool free = true;
    int samples_taken = 0;
    while(samples_taken < num_samples && free){
      distance += base_step * remap(float(samples_taken) / 10, 0, 1, 1, 3);
      free = CumuloNimbusGuaranteedFree(position + dir * distance);
      samples_taken = samples_taken + 1;
      if(!free){
        return last_distance;
      }
      last_distance = distance;
    }
    if(!CumuloNimbusGuaranteedFree(position + dir * distance)){
      return last_distance;
    }
    return distance;
  }
}

// mixture of HG, CS and Draine's phase proposed in "An Approximate Mie Scattering Function for Fog and Cloud Rendering"
float phaseComponent(float alpha, float g, float cos){
  return 1. / 4. / PI * (1 - pow(g, 2)) / pow(1 + pow(g, 2) - 2  * g * cos, 3/2.) * (1 + alpha * pow(cos, 2)) / (1 + alpha * (1 + 2 * pow(g, 2)) / 3.);
}

// phase function from "An Approximate Mie Scattering Function for Fog and Cloud Rendering"
float cloudPhase(vec3 r1, vec3 r2){
  float cos = dot(normalize(r1), normalize(-r2));
  // particle size parameter
  float d = 10e-6;
  float g_HG = exp(-0.0990567 / (d - 1.67154));
  float g_D = exp(-2.20679 / (d + 3.91029) - 0.428934);
  float alpha = exp(3.62489) - 8.29288 / (d + 5.52825);
  float omega_D = exp(-.599085 / (d - .641583)) - .665888;
  return (1 - omega_D) * phaseComponent(0, g_HG, cos) + omega_D * phaseComponent(alpha, g_D, cos);
}


// currently just a wrapper for one cloud density function
// this is the place for adding additional cloud bands
float getCloudDensity(vec3 position, vec3 cam_pos, bool hf = true){
  float acc = 0;
  float height = length(position) - PLANET_RADIUS;
  if(height > CUMULONIMBUS_START_HEIGHT && height < CUMULONIMBUS_END_HEIGHT){
    acc += getCumuloNimbusDensity(position, cam_pos, hf);
  }
  return acc;
}

float henyeyGreenstein(vec3 r1, vec3 r2, float g){
  float cosTheta = dot(normalize(r1), normalize(r2));
  float temp = 1 + pow(g, .5) + 2 * g * cosTheta;
  return (1 - g * g) / (temp * pow(temp, .5)) / 4 / PI;
  return 1 / 4 / PI * (1 - g * g) / pow(1 + g * g + 2 * g * cosTheta, 1.5);
}

// ray marching the transmittance through the cloud field. No adaptive step size used here
float raymarchTransmittance(vec3 rayOrigin, vec3 rayDir, vec2 interval, vec3 cam_pos, int samples=10){
  if(interval.y < 0){
    return 1.;
  }
  float t_last = interval.x;
  float path_transmittance = 1;
  float MAXIMUM_DIST_BETWEEN_SAMPLES = 1000;
  float interval_length = interval.y - interval.x;

  float last_extinction = getCloudDensity(rayOrigin, cam_pos) * DENSITY_MULTIPLIER * (1+ABSORBED_FRACTION);

  for(int i = 1; i <= samples; ++i){
    // more samples in the close vicinity of the point to capture self-shadowing of smaller clouds with fewer samples
    float progress = pow(float(i) / float(samples), 2);
    float t_now = remap(progress, 0, 1, interval.x, interval.y);

    float dist = t_now - t_last;
    vec3 position = rayOrigin + rayDir * t_now;
    float local_density = getCloudDensity(position, cam_pos, false);

    float scatter_coefficient = local_density * DENSITY_MULTIPLIER;
    float extinction = scatter_coefficient * (1+ABSORBED_FRACTION);
    float clamped_dist = clamp(dist, 0, MAXIMUM_DIST_BETWEEN_SAMPLES);
    float extinction_along_segment = exp(-(extinction + last_extinction) * .5 * clamped_dist);
    path_transmittance *= extinction_along_segment;
    last_extinction = extinction;
  }
  return path_transmittance;
}

// The function where all the integration happens
// uses adaptive step sizes to bring performance to an acceptable level
vec4 raymarchInterval(vec3 rayOrigin, vec3 rayDir, vec3 sunDir, vec2 interval, out vec3 path_transmittance, bool secondary_rays = false){
  if(interval.y < 0){
    path_transmittance = vec3(1);
    return vec4(0, 0, 0, 1);
  }

  // t values are parameters for rayOrigin + t * rayDir
  float t_last = interval.x;
  // progress is in [0, 1]
  float progress = 0;
  vec3 inscattering_acc = vec3(0.);
  path_transmittance = vec3(1);

  float interval_length = interval.y - interval.x;
  // tracked for debugging
  float skipped_distance = 0;

  // assuming the sun is infinitely far away, one phase calculation is enough because only single scattering is used
  float phase = cloudPhase(sunDir, -rayDir);
 
  vec3 atmo_transmittance;
  vec3 atmo_inscattering;

  // minimum normed progress that is made iteration to avoid crashes when changing step size logic
  float minimum_progress = 10. / interval_length;

  // track the start of the current cloud-free part of the interval. 
  // Inscattering from the atmosphere is added only when a cloud is encountered to reduce atmo model evaluations
  // DO NOT evaluate the atmosphere model for very short intervals
  float t_cloudfree_start = interval.x;
  float start_density = getCloudDensity(rayOrigin + rayDir * interval.x, rayOrigin);
  bool in_cloud = start_density > 0;
  float in_cloud_counter = 0;
  float last_scatter_coefficient = start_density * DENSITY_MULTIPLIER;
  int samples_taken = 0;
  float maximum_density = 0;

  //===== BEGIN OF RAY MARCHING LOOP ======
  
  while(progress < 1 && samples_taken < MAXIMUM_SAMPLES){
    samples_taken += 1;
    float t_now = remap(progress, 0, 1, interval.x, interval.y);
    vec3 position = rayOrigin + rayDir * t_now;

    //===== BEGIN OF STEP SIZE CONTROL =====

    // Skipping regions that are guaranteed to be free of clouds
    bool skipped = false;
    if(!in_cloud){
      float freeDistance = CumuloNimbusFreeDistance(position, rayDir, t_now, interval.y);
      if(freeDistance > 0){
        skipped_distance += freeDistance;
        progress += freeDistance / interval_length;
        skipped = true;
      }
    }
    
    // step size is increased when transmittance is low
    float low_transmittance_multiplier = remap(path_transmittance.r, .5, 0, 1, 3);
    // step size is increased when many samples have been taken already
    float samples_taken_multiplier = remap(float(samples_taken) / MAXIMUM_SAMPLES, 0, 1, 1, 10);
    // step size is decreased when the interval is short anyways
    float domain_length_multiplier = remap(interval_length, 10000, 30000, .5, 1);
    
    // there are two parameterized intervals with different functions for the step size as a function of distance from camera
    // Note that discontinuities in this function are a VERY BAD idea and give weird artifacts
    float step = CLOSE_STEP;
    if(t_now < MID_DISTANCE){
      step = remap(t_now, 0, MID_DISTANCE, CLOSE_STEP, MID_STEP);
    }else{
      step = remap(t_now, MID_DISTANCE, FAR_DISTANCE, MID_STEP, FAR_STEP);
    }
    step /= interval_length;
    progress += step * low_transmittance_multiplier * samples_taken_multiplier * domain_length_multiplier;
  
    progress = clamp(progress, 0, 1);
    t_now = remap(progress, 0, 1, interval.x, interval.y);
    float dist = t_now - t_last;
    position = rayOrigin + rayDir * t_now;

    //===== END OF STEP SIZE CONTROL =====
    //===== BEGIN OF SCATTERING INTEGRATION =====

    float local_density = getCloudDensity(position, rayOrigin);
    maximum_density = clamp(max(maximum_density, local_density), 0, 1);
    float scatter_coefficient = local_density * DENSITY_MULTIPLIER;
    // the light available at that point
    vec3 incoming_transmittance = vec3(1);
    vec3 local_incoming = GetSkyLuminance(position, sunDir, sunDir, incoming_transmittance);
    // using the luminance provided by the atmosphere model gives unstable results and artifacts
    // the transmittance from the atmosphere model seems fine though
    local_incoming = vec3(144809.5,129443.421875,127098.6484375) * incoming_transmittance;


    if(local_density > 0){
      //===== INSIDE CLOUD =====

      if(!in_cloud){
        // if entering cloud, integrate the inscattering from the regular atmosphere for the cloud-free interval
        atmo_inscattering = GetSkyLuminanceToPoint(rayOrigin + rayDir * t_cloudfree_start, position, sunDir, atmo_transmittance);
        inscattering_acc += path_transmittance * atmo_inscattering;
        path_transmittance *= atmo_transmittance;
      }
      in_cloud = true;
      in_cloud_counter +=1;
      t_cloudfree_start = t_now;

      // clamp to keep segment lengths reasonable to not break the lighting model
      float sdist = clamp(dist, 0, MAXIMUM_DIST_BETWEEN_SAMPLES);
      // extinction coefficient. 
      // Important to use midpoint for accurate Hillaire integration trick to work
      float sigma_e = (scatter_coefficient + last_scatter_coefficient) / 2 * (1 + ABSORBED_FRACTION);
      float transmittance_along_segment = exp(-sigma_e * sdist);
      vec3 direct_incoming = local_incoming;
    
      // get transmittance through clouds
      vec2 top_intersection = intersectSphere(position, sunDir, PLANET_RADIUS + CUMULONIMBUS_END_HEIGHT);
      int transmittance_samples = int(remap(path_transmittance.r, 0, 1, 10, 2));
      float in_transmittance = raymarchTransmittance(position, sunDir, vec2(0, top_intersection.y), rayOrigin, transmittance_samples);
      direct_incoming *= in_transmittance;

      // multiscattering approximation like in Nubis
      float ms_volume = remap(scatter_coefficient * dist, .1, 1.0, 0.0, 1.0);
      ms_volume *= pow(incoming_transmittance.r, 5);
      ms_volume *= MS_INTENSITY;
      vec3 msContrib = local_incoming * ms_volume;

      // Hillaire integration trick for more consistent appearance for different step sizes
      // considers the transmittance along the segment while integrating the scattering inside that segment
      // breaks when not using midpoint extinction coefficient
      vec3 S = scatter_coefficient * direct_incoming * phase * 4 * PI * CLOUD_COLOR + msContrib / sdist;
      inscattering_acc += path_transmittance * (S - S * transmittance_along_segment) / sigma_e;

      // reduce transmittance along the path
      path_transmittance *= vec3(transmittance_along_segment);
    }else{
      //===== NOT INSIDE CLOUD =====
      in_cloud = false;
      in_cloud_counter = 0;
    }

    //===== END OF SCATTERING INTEGRATION =====

    t_last = t_now;
    last_scatter_coefficient = scatter_coefficient;
    // terminate ray early when there is almost no transmittance left
    if (path_transmittance.r < .001){
      return vec4(inscattering_acc, path_transmittance.r);
    }
    // useful when working on adaptive step sizes to ensure that a step is always taken => fewer crashes during development
    progress += minimum_progress;
  }

  //===== END OF RAY MARCHING LOOP =====

  float skipped_fraction = skipped_distance / interval_length;
  //return vec4(skipped_fraction, 1-skipped_fraction, 0, 1)*10000;
  float sample_ratio = float(samples_taken) / float(MAXIMUM_SAMPLES);
  //return vec4(sample_ratio, 1-sample_ratio, 0, 1)*10000;
  //return vec4(maximum_density, 1-maximum_density, 0, 1)*10000;
  if(samples_taken == MAXIMUM_SAMPLES){
    return vec4(0, 0, 10000, 1);
  }

  // have to add atmo inscattering when exiting the interval
  if(!in_cloud){
    atmo_inscattering = GetSkyLuminanceToPoint(rayOrigin + rayDir * t_cloudfree_start, rayOrigin + rayDir * interval.y, sunDir, atmo_transmittance);
    inscattering_acc += path_transmittance * atmo_inscattering;
    path_transmittance *= atmo_transmittance;
  }

  // cheat a little near horizon: approximate scattering from surrounding atmosphere
  if(length(inscattering_acc) < 100 && path_transmittance.r < 1 - 1e-5){
    //surroundingColor = vec3(0, 10000, 0);
    path_transmittance = mix(vec3(1), path_transmittance, length(inscattering_acc) / 100);
  }
  return vec4(inscattering_acc, path_transmittance.r);// + mix(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), float(samples) / 100) * 100000;
}

// computes the cloud inscattered luminance in xyz and transmittance in alpha
vec4 getCloudColor(vec3 rayOrigin, vec3 rayDir, vec3 sunDir, float surfaceDistance, out vec3 transmittance) {
  float thickness = CUMULONIMBUS_END_HEIGHT - CUMULONIMBUS_START_HEIGHT;

  // The sphere radius of the upper-most and lowest cloud layers
  float topAltitude = PLANET_RADIUS + CUMULONIMBUS_END_HEIGHT;
  float lowAltitude = PLANET_RADIUS + CUMULONIMBUS_START_HEIGHT;

  vec2 topIntersections = intersectSphere(rayOrigin, rayDir, topAltitude);
  vec2 lowIntersections = intersectSphere(rayOrigin, rayDir, lowAltitude);
  vec2 atmo_intersections = intersectAtmosphere(rayOrigin, rayDir);

  float originHeight = length(rayOrigin);
  bool hitsSurface = surfaceDistance < atmo_intersections.y || intersectSphere(rayOrigin, rayDir, PLANET_RADIUS).y > 0;
  bool originInClouds = originHeight > lowAltitude && originHeight < topAltitude;

  bool hitTop = topIntersections.y > 0;
  bool hitBottom = lowIntersections.y > 0;

  bool above = originHeight > topAltitude;
  bool below = originHeight < lowAltitude;

  vec2 interval1 = vec2(0, -1);
  vec2 interval2 = vec2(0, -1);

  // use infintiy for no intersection to allow selecting other variables through min operation
  float lowXcorrected = lowIntersections.x < lowIntersections.y ? lowIntersections.x : INFINITY;

  if(above){
    interval1.x = topIntersections.x;
    interval1.y = lowIntersections.x;
    if(!hitsSurface){
      if(hitBottom){
        // ray exits the cloud layer at the bottom and reintersects it, creating a second interval
        interval2.x = lowIntersections.y;
        interval2.y = topIntersections.y;
      }else{
        // ray leaves the cloud layer on the upper side
        interval1.y = topIntersections.y;
      }
    }else{
      if(hitBottom){
        interval1.y = min(surfaceDistance, lowXcorrected);
      }else{
        interval1.y = surfaceDistance;
      }
      if(!hitTop || surfaceDistance < topIntersections.x){
        interval1.y = -1;
      }
    }
  }else{
    if(below){
      if(lowIntersections.y > 0){
        interval1.x = lowIntersections.y;
        interval1.y = topIntersections.y;
      }else{

      }
    }else{
      interval1.x = 0;
      if(lowIntersections.y > 0){
        interval1.y = lowIntersections.x;
        // check for second interval
        interval2.x = lowIntersections.y;
        interval2.y = topIntersections.y;
      }else{
        interval1.y = topIntersections.y;
      }
    }
  }

  if(interval1.y - interval1.x < 1){
    interval1.y = -1;
    interval1.x = 0;
  }

  interval1.y = min(interval1.y, surfaceDistance);
  interval2.y = min(interval2.y, surfaceDistance);

  if(interval1.y <= interval1.x){
    interval1.y = -1;
    interval1.x = 0;
  }

  //return vec4(interval1.y - interval1.x / 10000, 10000, 0, 1);

  if(interval2.y < interval2.x){
    interval2.y = -1;
    interval2.x = 0;
  }

  vec3 transmittance_int1 = vec3(1);
  vec3 transmittance_int2 = vec3(1);
  vec4 scatter_data1 = raymarchInterval(rayOrigin, rayDir, sunDir, interval1, transmittance_int1, true);
  vec4 scatter_data2 = vec4(0,0,0,1);
  if(scatter_data1.a > .0001){
    scatter_data2 = raymarchInterval(rayOrigin, rayDir, sunDir, interval2, transmittance_int2, true);
  }
  if(scatter_data1.x < 1e-6 && scatter_data2.x < 1e-6){
    // no significant inscattering from clouds. just return standard inscattering
    if(hitsSurface){
      return vec4(GetSkyLuminanceToPoint(rayOrigin, rayOrigin + surfaceDistance * rayDir, sunDir, transmittance), transmittance);
    }else{
      return vec4(GetSkyLuminance(rayOrigin, rayDir, sunDir, transmittance), transmittance.r);
    }
  }

  //return vec4(interval1.x, 1, 0, 1);
  vec3 transmittance_before_int1;
  vec3 inscattering_before_int1 = GetSkyLuminanceToPoint(rayOrigin, rayOrigin + rayDir * interval1.x, sunDir, transmittance_before_int1);

  if(scatter_data2.x < 1e-6){
    // no significant inscattering from second interval. return first interval inscattering with transmittance behind
    vec3 transmittance_behind_int1 = vec3(1);
    vec3 inscattering_behind_int1 = vec3(0);
    //return vec4(0, 1000, 0, 1);
    if(hitsSurface){
      if(surfaceDistance > interval1.y){
        inscattering_behind_int1 = GetSkyLuminanceToPoint(rayOrigin + rayDir * interval1.y, rayOrigin + rayDir * surfaceDistance, sunDir, transmittance_behind_int1);
      }
    }else{
      inscattering_behind_int1 = GetSkyLuminance(rayOrigin + rayDir * interval1.y, rayDir, sunDir, transmittance_behind_int1);
    }
    vec3 inScatter = inscattering_before_int1 + transmittance_before_int1 * (scatter_data1.xyz + transmittance_int1 * inscattering_behind_int1);
    transmittance = transmittance_before_int1 * transmittance_int1 * transmittance_behind_int1;
    return vec4(inScatter, transmittance.x);
  }else{
    vec3 transmittance_between_intervals = vec3(1);
    vec3 inscattering_between_intervals = GetSkyLuminanceToPoint(rayOrigin + rayDir * interval1.y, rayOrigin + rayDir * interval2.x, sunDir, transmittance_between_intervals);
    vec3 transmittance_behind = vec3(1);
    vec3 inscattering_behind = GetSkyLuminance(rayOrigin + rayDir * interval2.y, rayDir, sunDir, transmittance_behind);

    transmittance = transmittance_int1 * transmittance_between_intervals * transmittance_int2 * transmittance_behind;
    vec3 inScatter = inscattering_before_int1 + transmittance_before_int1 * (scatter_data1.xyz + transmittance_int1 * (inscattering_between_intervals + transmittance_between_intervals * (scatter_data2.xyz + transmittance_int2 * inscattering_behind)));
    return vec4(inScatter, transmittance.x);
  }

  return vec4(0, 100000, 0, 1);
}


// cloud density function for the standard cosmoscout cloud system
float getCloudDensityDefault(vec3 rayOrigin, vec3 rayDir, float tIntersection){
  vec3 position  = rayOrigin + rayDir * tIntersection;
  vec2 lngLat    = getLngLat(position);
  vec2 texCoords = vec2(lngLat.x / (2 * PI) + 0.5, 1.0 - lngLat.y / PI + 0.5);
  #if ENABLE_HDR
    return sRGBtoLinear(textureLod(uCloudTexture, texCoords, 0).r);
  #else
    return textureLod(uCloudTexture, texCoords, 0).r;
  #endif
}

// Computes the color of the clouds along the ray described by the input parameters. The cloud color
// is computed by intersecting 10 nested cloud layers. Each layer contributes a tenth of the final
// cloud density in order to create a fake volumetric appearance. The density is faded to zero close
// to mountains in order to prevent any hard seams.
// The color of the clouds is computed based on the sun and sky light which reaches the top layer of
// the clouds. Then, it is attenuated based on the transmittance of the atmosphere between the
// observer and the cloud.
// This method contains a couple of hard-coded values which could be made configurable in the
// future.
vec4 getCloudColorDefault(vec3 rayOrigin, vec3 rayDir, vec3 sunDir, float surfaceDistance){
// The distance between the top and bottom cloud layers.
  float thickness = uCloudAltitude * 0.5;

  // The distance to the planet surface where the fade-out starts.
  float fadeWidth = thickness * 2.0;

  // The altitude of the upper-most cloud layer.
  float topAltitude = PLANET_RADIUS + uCloudAltitude;

  // The number of cloud layers.
  int samples = 10;

  vec2 intersections = intersectSphere(rayOrigin, rayDir, topAltitude);

  // If we do not intersect the cloud sphere, we can return early.
  if (intersections.y < 0 || intersections.x > intersections.y) {
    return vec4(0.0);
  }

  // If we are below the clouds and the ray intersects the ground, we can also return early.
  if (intersections.x < 0 && surfaceDistance < intersections.y) {
    return vec4(0.0);
  }

  // Compute intersection point of view ray with clouds. Use this to compute the illuminance at this
  // point as well as the transmittance of the atmosphere towards the observer.
  vec3 p = rayOrigin + rayDir * (intersections.x < 0 ? intersections.y : intersections.x);
  vec3 skyIlluminance, transmittance;
  vec3 inScatter      = GetSkyLuminanceToPoint(rayOrigin, p, uSunDir, transmittance);
  vec3 sunIlluminance = GetSunAndSkyIlluminance(p, uSunDir, skyIlluminance);

  // We will accumulate the cloud density in this variable.
  float density = 0.0;

  for (int i = 0; i < samples; ++i) {
    float altitude      = topAltitude - i * thickness / samples;
    vec2  intersections = intersectSphere(rayOrigin, rayDir, altitude);
    float fac           = 1.0;

    // Reduce cloud opacity when end point is very close to planet surface.
    fac *= clamp(abs(surfaceDistance - intersections.x) / fadeWidth, 0, 1);
    fac *= clamp(abs(surfaceDistance - intersections.y) / fadeWidth, 0, 1);

    // Reduce cloud opacity when start point is very close to cloud surface.
    fac *= clamp(abs(intersections.x) / thickness, 0, 1);
    fac *= clamp(abs(intersections.y) / thickness, 0, 1);

    // If we intersect the cloud sphere...
    if (intersections.y > 0 && intersections.x < intersections.y) {

      // Check whether the cloud sphere is intersected from above...
      if (intersections.x > 0 && intersections.x < surfaceDistance) {
        // hits from above,
        density += getCloudDensityDefault(rayOrigin, rayDir, intersections.x) * fac;
      } else if (intersections.y < surfaceDistance) {
        // ... or from from below.
        density += getCloudDensityDefault(rayOrigin, rayDir, intersections.y) * fac;
      }
    }
  }

  // Compute the final color based on the cloud density.
  return vec4(
      transmittance * (sunIlluminance + skyIlluminance) / PI + inScatter, density / samples);
}

// This returns the density of the clouds when seen from rayOrigin looking into rayDir. This is used
// to compute the shadow of the clouds. To simplify the computation, only one cloud layer is
// assumed. If the ray's origin is close to the cloud layer (e.g. on a mountain), the density is
// faded to prevent any hard intersection seams. This contains a couple of hard-coded values which
// could be made configurable in the future.
float getCloudShadow(vec3 rayOrigin, vec3 rayDir) {
  float topAltitude   = PLANET_RADIUS + uCloudAltitude;
  vec2  intersections = intersectSphere(rayOrigin, rayDir, topAltitude);

  // If we do not intersect the cloud sphere, we can return early.
  if (intersections.y < 0 || intersections.x > intersections.y) {
    return 1.0;
  }

  // We have to fade out the cloud shadow in the same manner as the cloud color.
  float thickness = uCloudAltitude * 0.5;
  float fadeWidth = thickness * 2.0;

  // Reduce cloud opacity when end point is very close to planet surface.
  float fac = clamp(abs(intersections.y) / fadeWidth, 0, 1);
  vec3 position  = rayOrigin + rayDir * intersections.y;
  #if OLD_CLOUDS
  return 1.0 - getCloudDensityDefault(rayOrigin, rayDir, intersections.y) * fac;
  #else
  topAltitude = PLANET_RADIUS + CUMULONIMBUS_END_HEIGHT;
  thickness = CUMULONIMBUS_END_HEIGHT - CUMULONIMBUS_START_HEIGHT;

  vec2 topIntersections = intersectSphere(rayOrigin, rayDir, topAltitude);
  vec2 lowIntersections = intersectSphere(rayOrigin, rayDir, topAltitude - thickness);
  vec2 interval = vec2(lowIntersections.y, topIntersections.y);
  int transmittance_samples = int(remap(interval.y - interval.x, 10000, 100000, 5, 30));
  float transmittance = raymarchTransmittance(rayOrigin, rayDir, interval, rayOrigin, transmittance_samples);
  //float transmittance = clamp(raymarchingResult.a, .01, 1.); 
  return transmittance;
  #endif
}

// Returns a precomputed luminance of the atmosphere ring around the occluder for the
// given observer position and viewing direction. This is used if the observer is inside the
// shadow of the occluder. The normal atmosphere code would result in severe artifacts as only
// a few refracted sunrays would actually hit the observer. The precomputed texture has been
// generated by rendering the atmosphere from the occluder's point of view with a high resolution.
// The precomputed texture is stored in a small four-dimensional texture.
vec3 getApproximateLimbLuminance(vec3 rayOrigin, vec3 rayDir) {
  float dist     = length(rayOrigin);
  vec3  toCenter = rayOrigin / dist;
  vec3  projSun  = dot(uSunDir, toCenter) * toCenter - uSunDir;
  vec3  projAtmo = dot(rayDir, toCenter) * toCenter - rayDir;

  // The x and y coordinates of the texture are computed on the CPU and passed as a uniform. They
  // are based on the observer's position in the shadow volume.
  float x = uShadowCoordinates.x;
  float y = uShadowCoordinates.y;

  // For each [x, y] coordinate in the texture, the texture contains a two-dimensional image of the
  // atmosphere ring around the occluder as shown below. The image is stored layer-wise in the third
  // dimension of the texture. The first strip of pixels is the bottom layer, the second strip is
  // the second layer, and so on. The layers are stored consecutively in the texture. Usually, only
  // very few layers are needed.
  //
  //                    projSun
  //                       │        projAtmo
  // This part is          ┌┬--..  /
  // drawn below in  --->  ├┤-.  /'
  // more detail.          └┴-./ .  \
  //                       │β⁠/ \  .  │
  //                       o    │ .  │
  //                           /  .  │
  //                       ┌--'  .  /
  //                       ├ - '   .
  //                       └---''
  //
  float beta = acos(clamp(dot(normalize(projSun), normalize(projAtmo)), -1.0, 1.0)) / PI;

  // The texture is stored as a 3D texture with the size [res, res, layers * res].
  ivec3 texSize = textureSize(uLimbLuminanceTexture, 0);
  float res     = float(texSize.x);
  float layers  = float(texSize.z) / res;

  // If there is only one layer, we can use a fast path.
  if (layers == 1) {
    vec3 luminance = texture(uLimbLuminanceTexture, vec3(x, y, beta)).rgb;

#if !ENABLE_HDR
    luminance = tonemap(luminance / uSunInfo.y);
    luminance = linearToSRGB(luminance);
#endif

    return luminance;
  }

  // As shown above, the limb is vertically subdivided in a set of layers which are stored
  // consecutively in the texture. If there are two layers, the pixel strip from [x, y, 0] to
  // [x, y, 0.5] contains the luminance of the bottom layer, and the pixel strip from [x, y, 0.5] to
  // [x, y, 1.0] contains the luminance of the upper layer. The same applies for three layers, four
  // layers, and so on.

  // Add the layer start and end point we have to sample at pixel centers to avoid linear
  // interpolation with the start or end of the next or previous layer.
  float layerWidth = 1.0 / layers - 1.0 / res;

  float phiOcc      = asin(PLANET_RADIUS / dist);
  float phiAtmo     = asin(ATMOSPHERE_RADIUS / dist);
  float phi         = acos(clamp(dot(rayDir, -toCenter), -1.0, 1.0)) - phiOcc;
  float relativePhi = clamp(phi / (phiAtmo - phiOcc), 0.0, 1.0);

  // This is a visualization of a vertical cross-section of the atmosphere as shown above. The
  // planet is at the bottom, the upper atmosphere boundary is at the top. In this example, three
  // layers are used:
  //
  //   phiAtmo  ┌─────────┐ 1.0
  //            │         │
  //            │         │
  //            │         │
  //            ├─────────┤
  //            │         │
  //            │         │  relativePhi
  //            │         │
  //            ├─────────┤
  //            │         │
  //            │         │
  //            │         │
  //   phiOcc   └─────────┘ 0.0

  vec3 luminance;

  // In the upper half of the top layer and in the lower half of the bottom layer, we do not need to
  // blend between two layers. For all other positions, we blend between the two closest layers.
  if (relativePhi < 0.5 / layers || relativePhi > 1.0 - 0.5 / layers) {
    float layerStart = floor(relativePhi * layers) / layers + 0.5 / res;
    float z          = layerStart + beta * layerWidth;
    luminance        = texture(uLimbLuminanceTexture, vec3(x, y, z)).rgb;
  } else {
    float upperLayerStart = floor(relativePhi * layers + 0.5) / layers + 0.5 / res;
    float lowerLayerStart = floor(relativePhi * layers - 0.5) / layers + 0.5 / res;
    float upperZ          = upperLayerStart + beta * layerWidth;
    float lowerZ          = lowerLayerStart + beta * layerWidth;
    vec3  upperLuminance  = texture(uLimbLuminanceTexture, vec3(x, y, upperZ)).rgb;
    vec3  lowerLuminance  = texture(uLimbLuminanceTexture, vec3(x, y, lowerZ)).rgb;
    float blend           = relativePhi * layers - 0.5 - floor(relativePhi * layers - 0.5);
    luminance             = mix(lowerLuminance, upperLuminance, blend);
  }

#if !ENABLE_HDR
  luminance = tonemap(luminance / uSunInfo.y);
  luminance = linearToSRGB(luminance);
#endif

  return luminance;
}

// -------------------------------------------------------------------------------------------------

#if SKYDOME_MODE

uniform float uSunElevation;

// In this special mode, the atmosphere shader will draw a fish-eye view of the entire sky. This is
// meant for testing and debugging purposes.
void main() {
  float cSunElevation = uSunElevation / 180.0 * PI;

  // The altitude of the viewer above the ground.
  float cAltitude = 1.0;

  // If this is set to true, the horizon will be placed in the center of the fish-eye view.
  bool cHorizon = false;

  oColor = vec3(0.0);

  const vec3 rayOrigin = vec3(0.0, PLANET_RADIUS + cAltitude, 0.0);
  const vec3 sunDir    = normalize(vec3(0.0, sin(cSunElevation), cos(cSunElevation)));
  vec3       rayDir;
  vec2       coords = vsIn.texcoords * 2.0 - 1.0;

  if (length(coords) > 1.0) {
    return;
  }

  coords *= PI * 0.5;

  if (cHorizon) {
    rayDir = normalize(vec3(sin(coords.x), sin(coords.y), cos(length(coords))));

  } else {
    rayDir = normalize(vec3(sin(coords.x), cos(length(coords)), sin(-coords.y)));
  }

  vec2 atmosphereIntersections = intersectAtmosphere(rayOrigin, rayDir);
  if (atmosphereIntersections.x > atmosphereIntersections.y || atmosphereIntersections.y < 0) {
    return;
  }

  // If something is in front of the atmosphere, we do not have to do anything either.
  float surfaceDistance = getSurfaceDistance(rayOrigin, rayDir);
  if (surfaceDistance < atmosphereIntersections.x) {
    return;
  }

  bool hitsSurface = surfaceDistance < atmosphereIntersections.y;

  vec3 transmittance, inScatter;

  if (hitsSurface) {

    oColor = vec3(0.5);

    // If the ray hits the ground, we have to compute the amount of light reaching the ground as
    // well as how much of the reflected light is attenuated on its path to the observer.
    vec3 skyIlluminance;
    vec3 surfacePoint = rayOrigin + rayDir * surfaceDistance;
    inScatter         = GetSkyLuminanceToPoint(rayOrigin, surfacePoint, sunDir, transmittance);
    vec3 illuminance  = GetSunAndSkyIlluminance(surfacePoint, sunDir, skyIlluminance);
    illuminance += skyIlluminance;

    oColor *= illuminance;

  } else {
    oColor = vec3(0.0);

    // If the ray leaves the atmosphere unblocked, we only need to compute the luminance of the sky.
    inScatter = GetSkyLuminance(rayOrigin, rayDir, sunDir, transmittance);
  }

  oColor = transmittance * oColor + inScatter;
}

#else

// We start with the planet / background color without any atmosphere. First, we will overlay the
// water color (if enabled). Thereafter, we compute the atmospheric scattering and overlay the
// resulting atmosphere color. Finally, we will compute the color of the clouds and overlay them as
// well.
void main() {
  vec3 rayDir       = normalize(vsIn.rayDir);
  vec3 refractedRay = rayDir;

  // If the ray does not actually hit the atmosphere or the exit is already behind camera, we do not
  // have to modify the color any further.
  vec2 atmosphereIntersections = intersectAtmosphere(vsIn.rayOrigin, rayDir);
  if (atmosphereIntersections.x > atmosphereIntersections.y || atmosphereIntersections.y < 0) {
    oColor = getFramebufferColor(vsIn.texcoords);
    return;
  }

  // If something is in front of the atmosphere, we do not have to do anything either.
  float surfaceDistance = getSurfaceDistance(vsIn.rayOrigin, rayDir);
  if (surfaceDistance < atmosphereIntersections.x) {
    oColor = getFramebufferColor(vsIn.texcoords);
    return;
  }

  // If possible, use the precomputed limb luminance to get the color of the atmosphere ring around
  // the occluder.
#if ENABLE_LIMB_LUMINANCE
  if (RefractionSupported()) {
    // The third coordinate of the shadow coordinates is the approximate width of the atmosphere
    // ring around the occluder in pixels.
    float pixelWidth = uShadowCoordinates.z;

    // Use the limb luminance only if the observer is not too close to the occluder and if the
    // ring is thinner than 50 pixels.
    if (uShadowCoordinates.x > 0.05 && uShadowCoordinates.y > 0.0 && pixelWidth < 50.0) {

      vec2 planetIntersections = intersectPlanetsphere(vsIn.rayOrigin, rayDir);
      if (planetIntersections.x > planetIntersections.y) {
        oColor = getApproximateLimbLuminance(vsIn.rayOrigin, rayDir);
        return;
      }
    }
  }
#endif

  vec3 entryPoint =
      vsIn.rayOrigin + rayDir * (atmosphereIntersections.x > 0.0 ? atmosphereIntersections.x : 0.0);
  vec3 exitPoint =
      vsIn.rayOrigin + rayDir * (atmosphereIntersections.x > 0.0 ? atmosphereIntersections.x
                                                                 : atmosphereIntersections.y);

  // The ray hits an object if the distance to the depth buffer is smaller than the ray exit.
  bool hitsSurface = surfaceDistance < atmosphereIntersections.y;

  if (RefractionSupported() && !hitsSurface) {
    oColor = getRefractedFramebufferColor(entryPoint, rayDir, refractedRay);
  } else {
    oColor = getFramebufferColor(vsIn.texcoords);
  }

  // Always operate in linear color space.
#if !ENABLE_HDR
  oColor = sRGBtoLinear(oColor);
#endif

  bool underWater = false;

  vec4 oceanWaterShade   = vec4(0.0);
  vec4 oceanSurfaceColor = vec4(0.0);

#if ENABLE_WATER

  // We hit a water body if the ocean sphere is intersected and if the nearest surface is farther
  // away than the closest ocean sphere intersection (e.g. the ocean floor).
  vec2 oceanIntersections = intersectOceansphere(vsIn.rayOrigin, rayDir);
  bool hitsOcean = oceanIntersections.y > 0.0 && oceanIntersections.x < oceanIntersections.y;

  if (hitsOcean && surfaceDistance > oceanIntersections.x) {

    // Clamp the ray start to the camera position and the ray end to the ocean floor.
    oceanIntersections.x = max(oceanIntersections.x, 0);
    oceanIntersections.y = min(oceanIntersections.y, surfaceDistance);

    // Compute a water color based on the depth.
    float depth     = oceanIntersections.y - oceanIntersections.x;
    oceanWaterShade = getOceanShade(depth);

    // Looking down onto the ocean.
    if (oceanIntersections.x > 0) {
      vec3 oceanSurface = vsIn.rayOrigin + rayDir * oceanIntersections.x;
      vec3 idealNormal  = normalize(oceanSurface);

#if ENABLE_WAVES
      const float WAVE_SPEED        = 0.2;
      const float WAVE_SCALE        = 0.01;
      const float WAVE_STRENGTH     = 0.2;
      const float WAVE_MAX_DISTANCE = 10e4;

      // We compute three noise fields moving in three different directions.
      vec3 wave =
          vec3(simplex3DFractal(oceanSurface * WAVE_SCALE + vec3(0.0, 0.0, uTime * WAVE_SPEED)),
              simplex3DFractal(oceanSurface * WAVE_SCALE + vec3(0.0, uTime * WAVE_SPEED, 0.0)),
              simplex3DFractal(oceanSurface * WAVE_SCALE + vec3(uTime * WAVE_SPEED, 0.0, 0.0)));

      // The wave components are in [0...1]. With this we bring them into [-1...1] and apply some
      // smoothing.
      wave = sin((wave - 0.5) * PI);

      // As the waves produce a very noise aliasing pattern when seen from a large distance, we will
      // gradually hide them at large distances.
      float waveFade =
          pow(1 - clamp(distance(oceanSurface, vsIn.rayOrigin) / WAVE_MAX_DISTANCE, 0, 1), 4);

      // Intuitively, we should have accumulated all three noise fields to get a height field of the
      // waves. Then we should have computed the gradient of the height field to get the ocean
      // surface normal. However, computing the gradient is pretty expensive (dFdx and dFdy are not
      // precise enough). Hence, we simply use the 3D noise wave to modulate the ideal surface
      // normal. This is pretty hacky but results in a surprisingly wavy normal!
      vec3 normal = normalize(mix(idealNormal, wave, WAVE_STRENGTH * waveFade));
#else
      vec3  normal   = idealNormal;
      float waveFade = 0;
#endif

      // Now compute the reflected view ray. If this is reflected into the ocean sphere, we reflect
      // it once more up into the sky.
      vec3 reflection = reflect(rayDir, normal);
      if (dot(reflection, idealNormal) < 0.0) {
        reflection = reflect(reflection, idealNormal);
      }

      // Now get the sky color into the direction of the reflected view ray!
      vec3 transmittance;
      vec3 skyColor = GetSkyLuminance(oceanSurface, reflection, uSunDir, transmittance);

      // We later mix this into to ocean color using the Schlick approximation.
      float n           = 1.333;
      float r0          = pow(n / (1.0 + n), 2.0);
      float alpha       = r0 + (1.0 - r0) * pow(1.0 - clamp(dot(-rayDir, normal), 0.0, 1.0), 5.0);
      oceanSurfaceColor = vec4(skyColor, alpha * oceanWaterShade.a);

      // Now add a specular highlight for the Sun. This is not physically based but produces quite
      // pleasing results. If we are close to the waves, we use a hard specular, if we are farther
      // away we blend to a more soft specular which is supposed to look like the accumulation of
      // many small reflections of the Sun.
      float specularIntensity = clamp(dot(rayDir, reflect(uSunDir, normal)), 0, 1);
      float softSpecular      = pow(specularIntensity, 200) * 0.0001;
      float hardSpecular      = pow(specularIntensity, 2000) * 0.002;
      vec3  eclipseShadow     = getEclipseShadow((uMatM * vec4(oceanSurface, 1.0)).xyz);
      oceanSurfaceColor.rgb += mix(softSpecular, hardSpecular, waveFade) * uSunInfo.x *
                               transmittance * eclipseShadow *
                               pow(smoothstep(0, 1, dot(uSunDir, idealNormal)), 0.2);

#if !ENABLE_HDR
      // In non-HDR mode, we need to apply tone mapping to the ocean color.
      oceanSurfaceColor.rgb = tonemap(oceanSurfaceColor.rgb / uSunInfo.y);
#endif

      // The atmosphere now actually ends at the ocean surface.
      surfaceDistance = oceanIntersections.x;
      hitsSurface     = true;

    } else {
      underWater = true;
    }
  }
#endif

  vec3 eclipseShadow, transmittance, inScatter;

  // Retrieving the actual color contribution from the atmosphere is separated into two cases:
  // Either the ray hits something inside the atmosphere (e.g. we are looking towards the ground) or
  // it leaves the atmosphere unblocked (e.g. we are looking into the sky).
  if (hitsSurface) {

    // If the ray hits the ground, we have to compute the amount of light reaching the ground as
    // well as how much of the reflected light is attenuated on its path to the observer.
    vec3 skyIlluminance;
    vec3 surfacePoint = vsIn.rayOrigin + rayDir * surfaceDistance;
    inScatter        = GetSkyLuminanceToPoint(vsIn.rayOrigin, surfacePoint, uSunDir, transmittance);
    vec3 illuminance = GetSunAndSkyIlluminance(surfacePoint, uSunDir, skyIlluminance);
    illuminance += skyIlluminance;

    // The planetary surface should receive cloud shadows.
#if ENABLE_CLOUDS
    float cloudShadow = getCloudShadow(surfacePoint, uSunDir);
#else
    float cloudShadow = 1.0;
#endif

    // We also incorporate eclipse shadows. However, we only evaluate at the surface point. There is
    // no actual shadow volume in the atmosphere.
    eclipseShadow = getEclipseShadow((uMatM * vec4(surfacePoint, 1.0)).xyz);

    // We have to divide by uSunInfo.y because the planet shader already multiplied the result
    // of the BRDF with the Sun's illuminance. Since the planet shader does not know whether a
    // atmosphere will be drawn later, it only can assume that it is in direct sun light. However,
    // if there is an atmosphere, actually less light reaches the surface. So we have to divide by
    // the direct sun illuminance and multiply by the attenuated illuminance.
    oColor = cloudShadow * oColor * illuminance / uSunInfo.y;

    oceanSurfaceColor.rgb *= cloudShadow;

  } else {

    // If the ray leaves the atmosphere unblocked, we only need to compute the luminance of the sky.
    inScatter = GetSkyLuminance(vsIn.rayOrigin, rayDir, uSunDir, transmittance);

#if !ENABLE_HDR
    // If HDR Mode is disabled, we draw an artificial Sun. Else the Sun would look very dim close to
    // the horizon.
    if (angleBetweenVectors(refractedRay, uSunDir) < uSunInfo.z) {
      inScatter += uSunInfo.x;
    }
#endif

    // We also incorporate eclipse shadows. However, we only evaluate at the ray exit point. There
    // is no actual shadow volume in the atmosphere.
    eclipseShadow = getEclipseShadow((uMatM * vec4(exitPoint, 1.0)).xyz);
  }

  inScatter *= eclipseShadow;

#if !ENABLE_HDR
  inScatter = tonemap(inScatter / uSunInfo.y);
#endif

#if ENABLE_WATER
  if (!underWater) {
    oColor = mix(oColor, oColor * oceanWaterShade.rgb, oceanWaterShade.a);
    oColor = mix(oColor, oceanSurfaceColor.rgb, oceanSurfaceColor.a);
  }
#endif

  vec3 oColorOld = oColor;
  oColor = transmittance * oColor + inScatter;
  // save for compositing in new cloud model

#if ENABLE_WATER
  if (underWater) {
    oColor = mix(oColor, oColor * oceanWaterShade.rgb, oceanWaterShade.a);
  }
#endif

#if ENABLE_CLOUDS
#if OLD_CLOUDS
// Default cloud model, not truly physically based
  if (!underWater) {
    vec4 cloudColor = getCloudColorDefault(vsIn.rayOrigin, rayDir, uSunDir, surfaceDistance);
    cloudColor.rgb *= eclipseShadow;

#if !ENABLE_HDR
    cloudColor.rgb = tonemap(cloudColor.rgb / uSunInfo.y);
#endif

    oColor = mix(oColor, cloudColor.rgb, cloudColor.a);
  }
#else
  // new cloud model. Utilizes proper physically based rendering
  if (!underWater) {
    vec3 transmittance;
    vec4 cloudColor = getCloudColor(vsIn.rayOrigin, rayDir, uSunDir, surfaceDistance, transmittance);
    cloudColor.rgb *= eclipseShadow;
    

#if !ENABLE_HDR
    cloudColor.rgb = tonemap(cloudColor.rgb / uSunInfo.y);
#endif

    oColor = oColorOld * transmittance + cloudColor.rgb;
  }
#endif
#endif

// If HDR-mode is disabled, we have to convert to sRGB color space.
#if !ENABLE_HDR
  oColor = linearToSRGB(oColor);
#endif

  // This shouldn't happen. But if an atmospheric model produces a single NaN, the post processing
  // glare will make the entire screen black, so let's be rather safe then sorry :)
  if (any(isnan(oColor))) {
    oColor = vec3(0.0);
  }
}



#endif