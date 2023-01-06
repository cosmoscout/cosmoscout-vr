#version 330

// inputs
in VaryingStruct {
  vec3 vRayDir;
  vec3 vRayOrigin;
  vec2 vTexcoords;
}
vsIn;

// Returns the sky luminance along the segment from 'camera' to the nearest
// atmosphere boundary in direction 'view_ray', as well as the transmittance
// along this segment.
vec3 GetSkyLuminance(
    vec3 camera, vec3 view_ray, float shadow_length, vec3 sun_direction, out vec3 transmittance);

// Returns the sky luminance along the segment from 'camera' to 'p', as well as
// the transmittance along this segment.
vec3 GetSkyLuminanceToPoint(
    vec3 camera, vec3 p, float shadow_length, vec3 sun_direction, out vec3 transmittance);

// Returns the sun and sky illuminance received on a surface patch located at
// 'p' and whose normal vector is 'normal'.
vec3 GetSunAndSkyIlluminance(vec3 p, vec3 normal, vec3 sun_direction, out vec3 sky_illuminance);

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
uniform mat4  uMatInvMVP;
uniform float uWaterLevel;
// uniform mat4 uMatInvP;
// uniform mat4 uMatMV;
// uniform mat4 uMatM;
// uniform sampler2D uCloudTexture;
// uniform float     uCloudAltitude;

const float PI = 3.141592653589793;

// shadow stuff
// uniform sampler2DShadow uShadowMaps[5];
// uniform mat4 uShadowProjectionViewMatrices[5];
// uniform int  uShadowCascades;

// outputs
layout(location = 0) out vec3 oColor;

// ECLIPSE_SHADER_SNIPPET

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

// // for a given cascade and view space position, returns the lookup coordinates
// // for the corresponding shadow map
// vec3 GetShadowMapCoords(int cascade, vec3 position) {
//   vec4 smap_coords = uShadowProjectionViewMatrices[cascade] * vec4(position, 1.0);
//   return (smap_coords.xyz / smap_coords.w) * 0.5 + 0.5;
// }

// // returns the best cascade containing the given view space position
// int GetCascade(vec3 position) {
//   for (int i = 0; i < uShadowCascades; ++i) {
//     vec3 coords = GetShadowMapCoords(i, position);

//     if (coords.x > 0 && coords.x < 1 && coords.y > 0 && coords.y < 1 && coords.z > 0 &&
//         coords.z < 1) {
//       return i;
//     }
//   }

//   return -1;
// }

// // returns the amount of shadowing going on at the given view space position
// float GetShadow(vec3 position) {
//   int cascade = GetCascade(position);

//   if (cascade < 0) {
//     return 1.0;
//   }

//   vec3 coords = GetShadowMapCoords(cascade, position);

//   float shadow = 0;
//   float size   = 0.005;

//   for (int x = -1; x <= 1; x++) {
//     for (int y = -1; y <= 1; y++) {
//       vec2 off = vec2(x, y) * size;

//       // Dynamic array lookups are not supported in OpenGL 3.3
//       if (cascade == 0)
//         shadow += texture(uShadowMaps[0], coords - vec3(off, 0.00002));
//       else if (cascade == 1)
//         shadow += texture(uShadowMaps[1], coords - vec3(off, 0.00002));
//       else if (cascade == 2)
//         shadow += texture(uShadowMaps[2], coords - vec3(off, 0.00002));
//       else if (cascade == 3)
//         shadow += texture(uShadowMaps[3], coords - vec3(off, 0.00002));
//       else
//         shadow += texture(uShadowMaps[4], coords - vec3(off, 0.00002));
//     }
//   }

//   return shadow / 9.0;
// }

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
// current pixel, or -1 if there is nothing in the depth buffer
float getSurfaceDistance(vec3 rayOrigin, vec3 rayDir) {
  float depth = getDepth();

  if (depth == 0.0) {
    return -1.0;
  }

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
  if (surfaceDistance > 0.0 && surfaceDistance < atmosphereIntersections.x) {
    return;
  }

  bool hitsSurface = surfaceDistance > 0.0 && surfaceDistance < atmosphereIntersections.y;

#if ENABLE_WATER
  vec2 waterIntersections = intersectOceansphere(vsIn.vRayOrigin, rayDir);

  bool hitsOcean = waterIntersections.y > 0.0 && waterIntersections.x < waterIntersections.y;

  if (hitsOcean && (surfaceDistance < 0.0 || surfaceDistance > waterIntersections.x)) {
    if (surfaceDistance > 0.0) {
      waterIntersections.y = min(waterIntersections.y, surfaceDistance);
    }
    vec3  surface  = vsIn.vRayOrigin + rayDir * waterIntersections.x;
    vec3  normal   = normalize(surface);
    float specular = pow(max(dot(rayDir, reflect(uSunDir, normal)), 0.0), 10) * 0.2;
    specular += pow(max(dot(rayDir, reflect(uSunDir, normal)), 0.0), 50) * 0.2;
#if ENABLE_HDR
    specular *= uSunIlluminance / PI;
#endif
    float depth     = waterIntersections.y - waterIntersections.x;
    vec4  water     = getWaterShade(depth);
    oColor          = mix(oColor, water.rgb, water.a) + water.a * specular;
    surfaceDistance = waterIntersections.x;
    hitsSurface     = true;
  }
#endif

  float shadowLength = 0.0;

  if (hitsSurface) {
    vec3 skyIlluminance, transmittance;
    vec3 p = vsIn.vRayOrigin + rayDir * surfaceDistance;
    vec3 inScatter =
        GetSkyLuminanceToPoint(vsIn.vRayOrigin, p, shadowLength, uSunDir, transmittance);

    vec3 sunIlluminance = GetSunAndSkyIlluminance(p, normalize(p), uSunDir, skyIlluminance);

#if ENABLE_HDR
    oColor =
        transmittance * oColor / uSunIlluminance * (sunIlluminance + skyIlluminance) + inScatter;
#else
    oColor = linearToSRGB(transmittance * sRGBtoLinear(oColor) *
                              ((sunIlluminance + skyIlluminance) / uSunIlluminance) +
                          tonemap(inScatter / uSunIlluminance));
#endif
  } else {
    vec3 transmittance;
    vec3 inScatter = GetSkyLuminance(vsIn.vRayOrigin, rayDir, shadowLength, uSunDir, transmittance);

#if ENABLE_HDR
    oColor = transmittance * oColor + inScatter;
#else
    oColor =
        linearToSRGB(transmittance * sRGBtoLinear(oColor) + tonemap(inScatter / uSunIlluminance));
#endif
  }
}