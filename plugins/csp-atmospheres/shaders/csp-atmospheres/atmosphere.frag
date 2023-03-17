////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#version 330

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

uniform vec3  uSunDir;
uniform float uSunIlluminance;
uniform mat4 uMatM;
uniform mat4  uMatInvMVP;
uniform float uWaterLevel;
uniform sampler2D uCloudTexture;
uniform float     uCloudAltitude;

// outputs
layout(location = 0) out vec3 oColor;

// -------------------------------------------------------------------------------------------------

// Each atmospheric model will implement these three methods. We forward-declare them here. The
// actual implementation comes from the model's shader which is linked to this shader.

// Returns the sky luminance (in cd/m^2) along the segment from 'camera' to the nearest
// atmosphere boundary in direction 'viewRay', as well as the transmittance along this segment.
vec3 GetSkyLuminance(vec3 camera, vec3 viewRay, vec3 sunDirection, out vec3 transmittance);

// Returns the sky luminance (in cd/m^2) along the segment from 'camera' to 'p', as well as the
// transmittance along this segment.
vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 p, vec3 sunDirection, out vec3 transmittance);

// Returns the sun and sky illuminance (in lux) received on a surface patch located at 'p'.
vec3 GetSunAndSkyIlluminance(vec3 p, vec3 sunDirection, out vec3 skyIlluminance);

// -------------------------------------------------------------------------------------------------

// This will be replaced by the eclipse shdaer code.
// ECLIPSE_SHADER_SNIPPET

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

  if (position.z != 0.0) {
    result.x = atan(position.x / position.z);

    if (position.z < 0 && position.x < 0) {
      result.x -= PI;
    }

    if (position.z < 0 && position.x >= 0) {
      result.x += PI;
    }

  } else if (position.x == 0) {
    result.x = 0.0;
  } else if (position.x < 0) {
    result.x = -PI * 0.5;
  } else {
    result.x = PI * 0.5;
  }

  result.y = asin(position.y / length(position));
  return result;
}

// -------------------------------------------------------------------------------------------------

// Returns the background color at the current pixel. If multisampling is used, we take the average
// color.
vec3 getFramebufferColor() {
#if HDR_SAMPLES > 0
  vec3 color = vec3(0.0);
  for (int i = 0; i < HDR_SAMPLES; ++i) {
    color += texelFetch(uColorBuffer, ivec2(vsIn.texcoords * textureSize(uColorBuffer)), i).rgb;
  }
  return color / HDR_SAMPLES;
#else
  return texture(uColorBuffer, vsIn.texcoords).rgb;
#endif
}

// Returns the depth at the current pixel. If multisampling is used, we take the minimum depth.
float getFramebufferDepth() {
#if HDR_SAMPLES > 0
  float depth = 1.0;
  for (int i = 0; i < HDR_SAMPLES; ++i) {
    depth = min(
        depth, texelFetch(uDepthBuffer, ivec2(vsIn.texcoords * textureSize(uDepthBuffer)), i).r);
  }
  return depth;
#else
  return texture(uDepthBuffer, vsIn.texcoords).r;
#endif
}

// Returns the distance to the surface of the depth buffer at the current pixel. If the depth of the
// next opaque object is very close to the far end of our depth buffer, we will get jittering
// artifacts. That's the case if we are next to a satellite or on a moon and look towards a planet
// with an atmosphere. In this case, start and end of the ray through the atmosphere basically map
// to the same depth. Therefore, if the depth is really far away (close to zero) we compute the
// intersection with the planet analytically and blend to this value instead. This means, if you are
// close to a satellite, mountains of the planet below cannot poke through the atmosphere anymore.
float getSurfaceDistance(vec3 rayOrigin, vec3 rayDir) {
  float depth = getFramebufferDepth();

  // If the fragment is really far away, the inverse reverse infinite projection divides by zero.
  // So we add a minimum threshold here.
  depth = max(depth, 0.0000001);

  vec4  position = uMatInvMVP * vec4(2.0 * vsIn.texcoords - 1, 2 * depth - 1, 1);
  float depthMS  = length(rayOrigin - position.xyz / position.w);

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
vec4 getWaterShade(float d) {
  const float steps[5]  = float[](0.0, 50.0, 100.0, 500.0, 2000.0);
  const vec4  colors[5] = vec4[](vec4(1, 1, 1, 0.0), vec4(0.2, 0.8, 0.9, 0.0),
      vec4(0.2, 0.3, 0.4, 0.4), vec4(0.1, 0.2, 0.3, 0.8), vec4(0.03, 0.05, 0.1, 0.95));
  for (int i = 0; i < 4; ++i) {
    if (d <= steps[i + 1])
      return mix(colors[i], colors[i + 1], vec4(d - steps[i]) / (steps[i + 1] - steps[i]));
  }
  return colors[4];
}

// -------------------------------------------------------------------------------------------------

// Returns the value of the cloud texture at the position described by the three parameters.
float getCloudDensity(vec3 rayOrigin, vec3 rayDir, float tIntersection) {
  vec3 position  = rayOrigin + rayDir * tIntersection;
  vec2 lngLat    = getLngLat(position);
  vec2 texCoords = vec2(lngLat.x / (2 * PI) + 0.5, 1.0 - lngLat.y / PI + 0.5);
#if ENABLE_HDR
  return sRGBtoLinear(texture(uCloudTexture, texCoords).r);
#else
  return texture(uCloudTexture, texCoords).r;
#endif
}

// Computes the color of the clouds along the ray described by the input parameters. The cloud color
// is computed intersecting 10 nested cloud layers. Each layer contributes a tenth of the final
// cloud density in order to create a fake volumetric appearance. The density is faded to zero close
// to mountains in order to prevent any hard seams.
// The color of the clouds is computed based on the sun and sky light which reaches the top layer of
// the clouds. Then, it is attenuated based on the transmittance of the atmosphere between the
// observer and the cloud.
// This method contains a couple of hard-coded values which could be made configurable in the
// future.
vec4 getCloudColor(vec3 rayOrigin, vec3 rayDir, vec3 sunDir, float surfaceDistance) {

  // The distance between the top and bottom cloud layers.
  float thickness = uCloudAltitude * 0.2;

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
        density += getCloudDensity(rayOrigin, rayDir, intersections.x) * fac;
      } else if (intersections.y < surfaceDistance) {
        // ... or from from below.
        density += getCloudDensity(rayOrigin, rayDir, intersections.y) * fac;
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
  float thickness = uCloudAltitude * 0.2;
  float fadeWidth = thickness * 2.0;

  // Reduce cloud opacity when end point is very close to planet surface.
  float fac = clamp(abs(intersections.y) / fadeWidth, 0, 1);

  return 1.0 - getCloudDensity(rayOrigin, rayDir, intersections.y) * fac;
}

// -------------------------------------------------------------------------------------------------

// We start with the planet / background color without any atmosphere. First, we will overlay the
// water color (if enabled). Thereafter, we compute the atmospheric scattering and overlay the
// resulting atmosphere color. Finally, we will compute the color of the clouds and overlay them as
// well.
void main() {
  vec3 rayDir = normalize(vsIn.rayDir);

  // Get the planet / background color without any atmosphere.
  oColor = getFramebufferColor();

  // If the ray does not actually hit the atmosphere or the exit is already behind camera, we do not
  // have to modify the color any further.
  vec2 atmosphereIntersections = intersectAtmosphere(vsIn.rayOrigin, rayDir);
  if (atmosphereIntersections.x > atmosphereIntersections.y || atmosphereIntersections.y < 0) {
    return;
  }

  // If something is in front of the atmosphere, we do not have to do anything either.
  float surfaceDistance = getSurfaceDistance(vsIn.rayOrigin, rayDir);
  if (surfaceDistance < atmosphereIntersections.x) {
    return;
  }

  // The ray hits an object if the distance to the depth buffer is smaller than the ray exit.
  bool hitsSurface = surfaceDistance < atmosphereIntersections.y;

#if ENABLE_WATER

  // We hit a water body if the ocean sphere is intersected and if the nearest surface is farther
  // away than the closest ocean sphere intersection (e.g. the ocean floor).
  vec2 waterIntersections = intersectOceansphere(vsIn.rayOrigin, rayDir);
  bool hitsOcean = waterIntersections.y > 0.0 && waterIntersections.x < waterIntersections.y;

  if (hitsOcean && surfaceDistance > waterIntersections.x) {

    // Clamp the ray end to the ocean floor.
    waterIntersections.y = min(waterIntersections.y, surfaceDistance);

    // Compute a simple specular highlight.
    vec3  surfacePoint = vsIn.rayOrigin + rayDir * waterIntersections.x;
    vec3  normal       = normalize(surfacePoint);
    float specular     = pow(max(dot(rayDir, reflect(uSunDir, normal)), 0.0), 10) * 0.2;
    specular += pow(max(dot(rayDir, reflect(uSunDir, normal)), 0.0), 50) * 0.2;

    // In HDR-mode, we need to use the uSunIlluminance for the specular reflection.
#if ENABLE_HDR
    specular *= uSunIlluminance / PI;
#endif

    // Compute a water color based on the depth.
    float depth = waterIntersections.y - waterIntersections.x;
    vec4  water = getWaterShade(depth);

    // Also draw eclipse shadows onto the water surface.
    vec3 eclipseShadow = getEclipseShadow((uMatM * vec4(surfacePoint, 1.0)).xyz);
    oColor             = mix(oColor, water.rgb, water.a) + water.a * specular * eclipseShadow;

    // The atmosphere now actually end at the ocean surface.
    surfaceDistance = waterIntersections.x;
    hitsSurface     = true;
  }
#endif

  vec3 eclipseShadow;

  // Retrieving the actual color contribution from the atmosphere is separated into two cases:
  // Either the ray hits something inside the atmosphere (e.g. we are looking towards the ground) or
  // it leaves the atmosphere unblocked (e.g. we are looking into the sky).
  if (hitsSurface) {

    // If the ray hits the ground, we have to compute the amount of light reaching the ground as
    // well as how much of the reflected light is attenuated on its path to the observer.
    vec3 skyIlluminance, transmittance;
    vec3 surfacePoint = vsIn.rayOrigin + rayDir * surfaceDistance;
    vec3 inScatter   = GetSkyLuminanceToPoint(vsIn.rayOrigin, surfacePoint, uSunDir, transmittance);
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

    // We have to divide by uSunIlluminance because the planet shader already multiplied the result
    // of the BRDF with the Sun's illuminance. Since the planet shader does not know whether a
    // atmosphere will be drawn later, it only can assume that it is in direct sun light. However,
    // if there is an atmosphere, actually less light reaches the surface. So we have to divide by
    // the direct sun illuminance and multiply by the attenuated illuminance.
#if ENABLE_HDR
    oColor = transmittance * cloudShadow * oColor * illuminance / uSunIlluminance +
             inScatter * eclipseShadow;
#else
    oColor = transmittance * cloudShadow * sRGBtoLinear(oColor) * illuminance / uSunIlluminance +
             tonemap(eclipseShadow * inScatter / uSunIlluminance);
#endif

  } else {

    // If the ray leaves the atmosphere unblocked, we only need to compute the luminance of the sky.
    vec3 transmittance;
    vec3 inScatter = GetSkyLuminance(vsIn.rayOrigin, rayDir, uSunDir, transmittance);

    // We also incorporate eclipse shadows. However, we only evaluate at the ray exit point. There
    // is no actual shadow volume in the atmosphere.
    vec3 exitPoint =
        vsIn.rayOrigin + rayDir * (atmosphereIntersections.x > 0.0 ? atmosphereIntersections.x
                                                                   : atmosphereIntersections.y);
    eclipseShadow = getEclipseShadow((uMatM * vec4(exitPoint, 1.0)).xyz);

#if ENABLE_HDR
    oColor = transmittance * oColor + inScatter * eclipseShadow;
#else
    oColor =
        transmittance * sRGBtoLinear(oColor) + tonemap(inScatter * eclipseShadow / uSunIlluminance);
#endif
  }

// Last, but not least, add the clouds.
#if ENABLE_CLOUDS
  vec4 cloudColor = getCloudColor(vsIn.rayOrigin, rayDir, uSunDir, surfaceDistance);
  cloudColor.rgb *= eclipseShadow;

#if !ENABLE_HDR
  cloudColor.rgb = tonemap(cloudColor.rgb / uSunIlluminance);
#endif

  oColor = mix(oColor, cloudColor.rgb, cloudColor.a);
#endif

// If HDR-mode is disabled, we have to convert to sRGB color space.
#if !ENABLE_HDR
  oColor = linearToSRGB(oColor);
#endif
}