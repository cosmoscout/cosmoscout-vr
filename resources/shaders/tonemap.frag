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

// 4x4 bicubic filter using 4 bilinear texture lookups
// See GPU Gems 2: "Fast Third-Order Texture Filtering", Sigg & Hadwiger:
// http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter20.html

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
float w0(float a) {
  return (1.0 / 6.0) * (a * (a * (-a + 3.0) - 3.0) + 1.0);
}

float w1(float a) {
  return (1.0 / 6.0) * (a * a * (3.0 * a - 6.0) + 4.0);
}

float w2(float a) {
  return (1.0 / 6.0) * (a * (a * (-3.0 * a + 3.0) + 3.0) + 1.0);
}

float w3(float a) {
  return (1.0 / 6.0) * (a * a * a);
}

// g0 and g1 are the two amplitude functions
float g0(float a) {
  return w0(a) + w1(a);
}

float g1(float a) {
  return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
float h0(float a) {
  return -1.0 + w1(a) / (w0(a) + w1(a));
}

float h1(float a) {
  return 1.0 + w3(a) / (w2(a) + w3(a));
}

vec4 texture2D_bicubic(sampler2D tex, vec2 uv, int p_lod) {
  float lod        = float(p_lod);
  vec2  tex_size   = textureSize(uGlareMipMap, p_lod);
  vec2  pixel_size = 1.0 / tex_size;
  uv               = uv * tex_size + 0.5;
  vec2 iuv         = floor(uv);
  vec2 fuv         = fract(uv);

  float g0x = g0(fuv.x);
  float g1x = g1(fuv.x);
  float h0x = h0(fuv.x);
  float h1x = h1(fuv.x);
  float h0y = h0(fuv.y);
  float h1y = h1(fuv.y);

  vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - 0.5) * pixel_size;
  vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - 0.5) * pixel_size;
  vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - 0.5) * pixel_size;
  vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - 0.5) * pixel_size;

  return (g0(fuv.y) * (g0x * textureLod(tex, p0, lod) + g1x * textureLod(tex, p1, lod))) +
         (g1(fuv.y) * (g0x * textureLod(tex, p2, lod) + g1x * textureLod(tex, p3, lod)));
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

  if (uGlareIntensity > 0) {
#ifdef BICUBIC_GLARE_FILTER

      vec3 glare = texture2D(uGlareMipMap, vTexcoords, 0).rgb;

      color = mix(color, glare, pow(uGlareIntensity, 2.0));
#else


    vec3  glare     = vec3(0);
    float maxLevels = textureQueryLevels(uGlareMipMap);

    float totalWeight = 0;

    // Each level contains a successively more blurred version of the scene. We have to
    // accumulate them with an exponentially decreasing weight to get a proper glare distribution.
    for (int i = 0; i < maxLevels; ++i) {
      float weight = 1.0 / pow(2, i);


      glare += texture2D(uGlareMipMap, vTexcoords, i).rgb * weight;


      totalWeight += weight;
    }

    // To make sure that we do not add energy, we divide by the total weight.
    color = mix(color, glare / totalWeight, pow(uGlareIntensity, 2.0));
#endif

  }

// Filmic
#if TONE_MAPPING_MODE == 2
  color           = Uncharted2Tonemap(uExposure * color);
  vec3 whiteScale = vec3(1.0) / Uncharted2Tonemap(vec3(W));
  oColor          = linear_to_srgb(color * whiteScale);

// Gamma only
#elif TONE_MAPPING_MODE == 1
  oColor = linear_to_srgb(uExposure * color);

// None
#else
  oColor = uExposure * color;
#endif
}