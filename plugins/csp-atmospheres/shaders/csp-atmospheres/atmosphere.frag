////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#version 330

// inputs
in VaryingStruct {
  vec3 vRayDir;
  vec3 vRayOrigin;
  vec2 vTexcoords;
}
vsIn;

// Returns the sky luminance along the segment from 'camera' to the nearest
// atmosphere boundary in direction 'viewRay', as well as the transmittance
// along this segment.
vec3 GetSkyLuminance(vec3 camera, vec3 viewRay, vec3 sunDirection, out vec3 transmittance);

// Returns the sky luminance along the segment from 'camera' to 'p', as well as
// the transmittance along this segment.
vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 p, vec3 sunDirection, out vec3 transmittance);

// Returns the sun and sky illuminance received on a surface patch located at
// 'p'.
vec3 GetSunAndSkyIlluminance(vec3 p, vec3 sunDirection, out vec3 skyIlluminance);

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

const float PI = 3.141592653589793;

// outputs
layout(location = 0) out vec3 oColor;

// ECLIPSE_SHADER_SNIPPET

// -------------------------------------------------------------------------------------------------

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

// compute intersections with the atmosphere
// two T parameters are returned -- if no intersection is found, the first will
// larger than the second
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

vec2 intersectAtmosphere(vec3 rayOrigin, vec3 rayDir) {
  return intersectSphere(rayOrigin, rayDir, ATMOSPHERE_RADIUS);
}

vec2 intersectPlanetsphere(vec3 rayOrigin, vec3 rayDir) {
  return intersectSphere(rayOrigin, rayDir, PLANET_RADIUS);
}

vec2 intersectOceansphere(vec3 rayOrigin, vec3 rayDir) {
  return intersectSphere(rayOrigin, rayDir, uWaterLevel + PLANET_RADIUS);
}

// Returns the depth at the current pixel. If multisampling is used, we take the minimum depth.
float getDepth() {
#if HDR_SAMPLES > 0
  float depth = 1.0;
  for (int i = 0; i < HDR_SAMPLES; ++i) {
    depth = min(
        depth, texelFetch(uDepthBuffer, ivec2(vsIn.vTexcoords * textureSize(uDepthBuffer)), i).r);
  }
  return depth;
#else
  return texture(uDepthBuffer, vsIn.vTexcoords).r;
#endif
}

// returns the model space distance to the surface of the depth buffer at the
// current pixel
float getSurfaceDistance(vec3 rayOrigin, vec3 rayDir) {
  float depth = getDepth();

  // If the fragment is really far away, the inverse reverse infinite projection divides by zero.
  // So we add a minimum threshold here.
  depth = max(depth, 0.0000001);

  vec4  position = uMatInvMVP * vec4(2.0 * vsIn.vTexcoords - 1, 2 * depth - 1, 1);
  float depthMS  = length(rayOrigin - position.xyz / position.w);

  // If the depth of the next opaque object is verz close to the far end of our depth buffer, we
  // will get jittering artifacts. That's the case if we are next to a satellite or on a moon and
  // look towards a planet with an atmosphere. In this case, start and end of the ray through the
  // atmosphere basically map to the same depth. Therefore, if the depth is really far away (close
  // to zero) we compute the intersection with the planet analytically and blend to this value
  // instead. This means, if you are close to a satellite, mountains of the planet below cannot
  // poke through the atmosphere anymore.
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

// Returns the background color at the current pixel. If multisampling is used, we take the average
// color.
vec3 getBackgroundColor() {
#if HDR_SAMPLES > 0
  vec3 color = vec3(0.0);
  for (int i = 0; i < HDR_SAMPLES; ++i) {
    color += texelFetch(uColorBuffer, ivec2(vsIn.vTexcoords * textureSize(uColorBuffer)), i).rgb;
  }
  return color / HDR_SAMPLES;
#else
  return texture(uColorBuffer, vsIn.vTexcoords).rgb;
#endif
}

// -------------------------------------------------------------------------------------------------

// returns a hard-coded color scale for a given ocean depth.
// Could be configurable in future.
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

  // geocentric latitude of the input point
  result.y = asin(position.y / length(position));
  return result;
}

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

vec4 getCloudColor(vec3 rayOrigin, vec3 rayDir, vec3 sunDir, float surfaceDistance) {
  float density   = 0.0;
  float thickness = uCloudAltitude * 0.2;
  float fadeWidth = thickness * 2.0;
  float height    = PLANET_RADIUS + uCloudAltitude;
  int   samples   = 10;

  vec2 intersections = intersectSphere(rayOrigin, rayDir, height);

  // If we do not intersect the cloud sphere...
  if (intersections.y < 0 || intersections.x > intersections.y) {
    return vec4(0.0);
  }

  vec3 p = rayOrigin + rayDir * (intersections.x < 0 ? intersections.y : intersections.x);
  vec3 skyIlluminance, transmittance;
  vec3 inScatter      = GetSkyLuminanceToPoint(rayOrigin, p, uSunDir, transmittance);
  vec3 sunIlluminance = GetSunAndSkyIlluminance(p, uSunDir, skyIlluminance);

  for (int i = 0; i < samples; ++i) {
    float altitude      = height - i * thickness / samples;
    vec2  intersections = intersectSphere(rayOrigin, rayDir, altitude);
    float fac           = 1.0;

    // reduce cloud opacity when end point is very close to planet surface
    fac *= clamp(abs(surfaceDistance - intersections.x) / fadeWidth, 0, 1);
    fac *= clamp(abs(surfaceDistance - intersections.y) / fadeWidth, 0, 1);

    // reduce cloud opacity when start point is very close to cloud surface
    fac *= clamp(abs(intersections.x) / thickness, 0, 1);
    fac *= clamp(abs(intersections.y) / thickness, 0, 1);

    // If we intersect the cloud sphere...
    if (intersections.y > 0 && intersections.x < intersections.y) {

      // Check whether the cloud sphere is intersected from above...
      if (intersections.x > 0 && intersections.x < surfaceDistance) {
        // hits from above
        density += getCloudDensity(rayOrigin, rayDir, intersections.x) * fac;
      } else if (intersections.y < surfaceDistance) {
        // ... or from from below
        density += getCloudDensity(rayOrigin, rayDir, intersections.y) * fac;
      }
    }
  }

  return vec4(
      transmittance * (sunIlluminance + skyIlluminance) / PI + inScatter, density / samples);
}

float getCloudShadow(vec3 rayOrigin, vec3 rayDir) {
  float altitude      = PLANET_RADIUS + uCloudAltitude;
  vec2  intersections = intersectSphere(rayOrigin, rayDir, altitude);
  float thickness     = uCloudAltitude * 0.2;
  float fadeWidth     = thickness * 2.0;
  float fac           = 1.0;

  // reduce cloud opacity when end point is very close to planet surface
  // fac *= clamp(abs(intersections.x) / fadeWidth, 0, 1);

  if (intersections.y > 0 && intersections.x < intersections.y) {
    return 1.0 - getCloudDensity(rayOrigin, rayDir, intersections.y) * fac;
  }

  return 1.0;
}

// -------------------------------------------------------------------------------------------------

void main() {
  vec3 rayDir = normalize(vsIn.vRayDir);

  oColor = getBackgroundColor();

  vec2 atmosphereIntersections = intersectAtmosphere(vsIn.vRayOrigin, rayDir);

  // ray does not actually hit the atmosphere or exit is behind camera
  if (atmosphereIntersections.x > atmosphereIntersections.y || atmosphereIntersections.y < 0) {
    return;
  }

  float surfaceDistance = getSurfaceDistance(vsIn.vRayOrigin, rayDir);

  // something is in front of the atmosphere
  if (surfaceDistance < atmosphereIntersections.x) {
    return;
  }

  bool hitsSurface = surfaceDistance < atmosphereIntersections.y;

  vec3 eclipseShadow;

#if ENABLE_WATER
  vec2 waterIntersections = intersectOceansphere(vsIn.vRayOrigin, rayDir);

  bool hitsOcean = waterIntersections.y > 0.0 && waterIntersections.x < waterIntersections.y;

  if (hitsOcean && (surfaceDistance < 0.0 || surfaceDistance > waterIntersections.x)) {
    if (surfaceDistance > 0.0) {
      waterIntersections.y = min(waterIntersections.y, surfaceDistance);
    }

    vec3 p        = vsIn.vRayOrigin + rayDir * waterIntersections.x;
    eclipseShadow = getEclipseShadow((uMatM * vec4(p, 1.0)).xyz);

    vec3  normal   = normalize(p);
    float specular = pow(max(dot(rayDir, reflect(uSunDir, normal)), 0.0), 10) * 0.2;
    specular += pow(max(dot(rayDir, reflect(uSunDir, normal)), 0.0), 50) * 0.2;
#if ENABLE_HDR
    specular *= uSunIlluminance / PI;
#endif
    float depth     = waterIntersections.y - waterIntersections.x;
    vec4  water     = getWaterShade(depth);
    oColor          = mix(oColor, water.rgb, water.a) + water.a * specular * eclipseShadow;
    surfaceDistance = waterIntersections.x;
    hitsSurface     = true;
  }
#endif

  if (hitsSurface) {
    vec3 skyIlluminance, transmittance;
    vec3 p         = vsIn.vRayOrigin + rayDir * surfaceDistance;
    vec3 inScatter = GetSkyLuminanceToPoint(vsIn.vRayOrigin, p, uSunDir, transmittance);
    eclipseShadow  = getEclipseShadow((uMatM * vec4(p, 1.0)).xyz);

    vec3 illuminance = GetSunAndSkyIlluminance(p, uSunDir, skyIlluminance);
    illuminance += skyIlluminance;

#if ENABLE_CLOUDS
    float cloudShadow = getCloudShadow(p, uSunDir);
#else
    float cloudShadow = 1.0;
#endif

#if ENABLE_HDR
    oColor = transmittance * cloudShadow * oColor / uSunIlluminance * illuminance +
             inScatter * eclipseShadow;
#else
    oColor = transmittance * cloudShadow * sRGBtoLinear(oColor) * illuminance / uSunIlluminance +
             tonemap(eclipseShadow * inScatter / uSunIlluminance);
#endif
  } else {
    vec3 transmittance;
    vec3 inScatter = GetSkyLuminance(vsIn.vRayOrigin, rayDir, uSunDir, transmittance);

    vec3 p =
        vsIn.vRayOrigin + rayDir * (atmosphereIntersections.x > 0.0 ? atmosphereIntersections.x
                                                                    : atmosphereIntersections.y);
    eclipseShadow = getEclipseShadow((uMatM * vec4(p, 1.0)).xyz);

#if ENABLE_HDR
    oColor = transmittance * oColor + inScatter * eclipseShadow;
#else
    oColor =
        transmittance * sRGBtoLinear(oColor) + tonemap(inScatter * eclipseShadow / uSunIlluminance);
#endif
  }

#if ENABLE_CLOUDS
  vec4 cloudColor = getCloudColor(vsIn.vRayOrigin, rayDir, uSunDir, surfaceDistance);
  cloudColor.rgb *= eclipseShadow;

#if !ENABLE_HDR
  cloudColor.rgb = tonemap(cloudColor.rgb / uSunIlluminance);
#endif

  oColor = mix(oColor, cloudColor.rgb, cloudColor.a);
#endif

#if !ENABLE_HDR
  oColor = linearToSRGB(oColor);
#endif
}