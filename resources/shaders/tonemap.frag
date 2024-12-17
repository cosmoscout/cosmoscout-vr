////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

in vec2 vTexcoords;

layout(pixel_center_integer) in vec4 gl_FragCoord;

#if NUM_MULTISAMPLES > 0
layout(binding = 0) uniform sampler2DMS uComposite;
layout(binding = 1) uniform sampler2DMS uDepth;
#else
layout(binding = 0) uniform sampler2D uComposite;
layout(binding = 1) uniform sampler2D uDepth;
#endif

layout(binding = 2) uniform sampler2D uGlareMipMap;

uniform float uExposure;
uniform float uMaxLuminance;
uniform float uGlareIntensity;

layout(location = 0) out vec3 oColor;

// http://filmicworlds.com/blog/filmic-tonemapping-operators/
float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 Uncharted2Tonemap(vec3 x) {
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

float linear_to_srgb(float c) {
  if (c <= 0.0031308)
    return 12.92 * c;
  else
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

vec3 linear_to_srgb(vec3 c) {
  return vec3(linear_to_srgb(c.r), linear_to_srgb(c.g), linear_to_srgb(c.b));
}

void main() {
#if NUM_MULTISAMPLES > 0
  vec3 color = vec3(0.0);
  for (int i = 0; i < NUM_MULTISAMPLES; ++i) {
    color += texelFetch(uComposite, ivec2(vTexcoords * textureSize(uComposite)), i).rgb;
  }
  color /= NUM_MULTISAMPLES;

  float depth = 1.0;
  for (int i = 0; i < NUM_MULTISAMPLES; ++i) {
    depth = min(depth, texelFetch(uDepth, ivec2(vTexcoords * textureSize(uDepth)), i).r);
  }
  gl_FragDepth = depth;
#else
  vec3 color   = texelFetch(uComposite, ivec2(vTexcoords * textureSize(uComposite, 0)), 0).rgb;
  gl_FragDepth = texelFetch(uDepth, ivec2(vTexcoords * textureSize(uDepth, 0)), 0).r;
#endif

  // Glare is scaled by max luminance.
  if (uGlareIntensity > 0) {
    vec3 glare = texture(uGlareMipMap, vTexcoords, 0).rgb * uMaxLuminance / 65500;
    color      = mix(color, glare, pow(uGlareIntensity, 2.0));
  }

  color *= uExposure;

// Filmic
#if TONE_MAPPING_MODE == 2
  color           = Uncharted2Tonemap(color);
  vec3 whiteScale = vec3(1.0) / Uncharted2Tonemap(vec3(W));
  oColor          = linear_to_srgb(color * whiteScale);

// Gamma only
#elif TONE_MAPPING_MODE == 1
  oColor = linear_to_srgb(color);

// None
#else
  oColor = color;
#endif
}