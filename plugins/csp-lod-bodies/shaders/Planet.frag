#version 430

////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#define SHOW_TEXTURE_RGB 0

uniform float heightMin;
uniform float heightMax;
uniform float slopeMin;
uniform float slopeMax;
uniform float ambientBrightness;
uniform float ambientOcclusion;
uniform float texGamma;
uniform vec4  uSunDirIlluminance;

uniform sampler1D heightTex;
uniform sampler2D fontTex;

// ==========================================================================
// include eclipse shadow computation code
$ECLIPSE_SHADER_SNIPPET

// ==========================================================================
// include helper functions/declarations from VistaPlanet
$VP_TERRAIN_SHADER_UNIFORMS
$VP_TERRAIN_SHADER_FUNCTIONS

// inputs
// ==========================================================================

in VS_OUT {
  vec2  tileCoords;
  vec3  normal;
  vec3  position;
  vec3  planetCenter;
  vec2  lngLat;
  float height;
  vec2  vertexPosition;
  vec3  sunDir;
}
fsIn;

vec3 heat(float v) {
  float value = 1.0 - v;
  return (0.5 + 0.5 * smoothstep(0.0, 0.1, value)) *
         vec3(smoothstep(0.5, 0.3, value),
             value < 0.3 ? smoothstep(0.0, 0.3, value) : smoothstep(1.0, 0.6, value),
             smoothstep(0.4, 0.6, value));
}

// outputs
// ==========================================================================

layout(location = 0) out vec4 fragColor;

vec3 SRGBtoLINEAR(vec3 srgbIn) {
  vec3 bLess = step(vec3(0.04045), srgbIn);
  return mix(srgbIn / vec3(12.92), pow((srgbIn + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
}

// placeholder for the BRDF in HDR mode
$BRDF_HDR

// placeholder for the BRDF in light mode
$BRDF_NON_HDR

void main() {
  if (VP_shadowMapMode) {
    return;
  }

  fragColor = vec4(1);

  vec3 idealNormal = normalize(fsIn.position - fsIn.planetCenter);

#if $LIGHTING_QUALITY > 1
  vec3 surfaceNormal = normalize(fsIn.normal);
#else
  vec3 dx            = dFdx(fsIn.position);
  vec3 dy            = dFdy(fsIn.position);
  vec3 surfaceNormal = normalize(cross(dx, dy));
#endif

#if $SHOW_TEXTURE
  // Make sure to sample at the pixel centers.
  float pixelSize = 1.0 / VP_getResolutionIMG();
  vec2 texcoords = fsIn.tileCoords * (1.0 - pixelSize) + 0.5 * pixelSize;
  fragColor.rgb = texture(VP_texIMG, vec3(texcoords, VP_dataLayers.y)).rgb;

#if $ENABLE_HDR
  fragColor.rgb = SRGBtoLINEAR(fragColor.rgb);
#endif

  fragColor.rgb = pow(fragColor.rgb, vec3(1.0 / texGamma));
#endif

#if $COLOR_MAPPING_TYPE == 1
  {
    float height      = clamp(fsIn.height, heightMin, heightMax);
    float height_norm = (height - heightMin) / (heightMax - heightMin);
    fragColor *= texture(heightTex, height_norm);
  }
#endif

#if $COLOR_MAPPING_TYPE == 2
  {
    float slope = acos(dot(idealNormal, surfaceNormal));
    float fac   = clamp((slope - slopeMin) / (slopeMax - slopeMin), 0.0, 1.0);
    fragColor *= texture(heightTex, fac);
  }
#endif

  // Needed for the BRDFs.
  vec3  N     = normalize(surfaceNormal);
  vec3  L     = normalize(fsIn.sunDir);
  vec3  V     = normalize(-fsIn.position);
  float cos_i = dot(N, L);
  float cos_r = dot(N, V);

  vec3 luminance = vec3(1.0);
#if $ENABLE_SHADOWS
  luminance *= VP_getShadow(fsIn.position);
#endif

#if $ENABLE_HDR
  // Make the amount of ambient brightness perceptually linear in HDR mode.
  float ambient = pow(ambientBrightness, VP_E);
  float f_r = BRDF_HDR(N, L, V);
#else
  float ambient = ambientBrightness;
  float f_r = BRDF_NON_HDR(N, L, V);
#endif

#if $ENABLE_LIGHTING
  luminance *= max(0.0, cos_i);

  if (cos_i > 0) {
    if (f_r < 0 || isnan(f_r) || isinf(f_r)) {
      luminance *= 0;
    } else {
      luminance *= f_r * getEclipseShadow(fsIn.position);
    }
  }
  fragColor.rgb = mix(fragColor.rgb * luminance, fragColor.rgb, ambient);

  // Add some hill shading (pseudo ambient occlusion).
  fragColor.rgb *= mix(1.0, max(0, dot(idealNormal, surfaceNormal)), ambientOcclusion);
#endif

#if $ENABLE_HDR
  // In HDR-mode, we have to add the sun's luminance and divide by the average intensity of the
  // texture map.
  fragColor.rgb *= uSunDirIlluminance.w / $AVG_LINEAR_IMG_INTENSITY;
#endif

// conserve energy
#if $ENABLE_HDR && !$ENABLE_LIGHTING
  fragColor.rgb /= VP_PI;
#endif

#if $SHOW_TILE_BORDER
  // color area by level
  const float minLevel   = 1;
  const float maxLevel   = 15;
  const float brightness = 0.3;
  const float alpha      = 0.5;

  float level      = clamp(log2(float(VP_offsetScale.z)), minLevel, maxLevel);
  vec4  debugColor = vec4(heat((level - minLevel) / (maxLevel - minLevel)), alpha);
  debugColor.rgb   = mix(debugColor.rgb, vec3(1), brightness);

  // Create a red border around each tile. As the outer-most vertex is the bottom of the skirt, we
  // have to make the border 1.5 pixels wide to be visible on the top of the tile.
  float edgeWidth = 1.5;
  int   maxVertex = VP_getResolutionDEM() + 1;

  if (fsIn.vertexPosition.x < edgeWidth || fsIn.vertexPosition.y < edgeWidth ||
      fsIn.vertexPosition.x > maxVertex - edgeWidth || 
      fsIn.vertexPosition.y > maxVertex - edgeWidth) {
    debugColor = vec4(1.0, 0.0, 0.0, alpha);
  }

#if $ENABLE_HDR
  // Make sure that the color overlays are visible in HDR mode.
  debugColor.rgb = SRGBtoLINEAR(debugColor.rgb);
  debugColor.rgb *= uSunDirIlluminance.w / VP_PI / $AVG_LINEAR_IMG_INTENSITY;
#endif

  fragColor.rgb = mix(fragColor.rgb, debugColor.rgb, debugColor.a);
#endif

#if $ENABLE_SHADOWS_DEBUG && $ENABLE_SHADOWS
  float cascade = VP_getCascade(fsIn.position);
  if (cascade >= 0) {
    fragColor.rgb = mix(fragColor.rgb, heat(1 - cascade / (VP_shadowCascades - 1)), 0.2);
  }
#endif

  vec3  viewDir    = -fsIn.position;
  float camDist    = length(viewDir);
  float centerDist = length(fsIn.planetCenter);

#if $SHOW_LAT_LONG || $SHOW_LAT_LONG_LABELS
  {
#if $ENABLE_LIGHTING
    float fIdealLightIntensity =
        dot(idealNormal, normalize(fsIn.sunDir)) * (1.0 - ambient) + ambient;
    vec3 grid_color =
        mix(fragColor.rgb, vec3(mix(1.0, 0.0, clamp(fIdealLightIntensity + 1.0, 0.0, 1.0))), 0.8);
#else
    vec3 grid_color = mix(fragColor.rgb, vec3(0), 0.8);
#endif

    const float spacings[4]        = float[](500.0, 50.0, 5.0, 1.0);
    const float distances[4]       = float[](1.5, 0.8, 0.4, 0.1);
    const float intensities[4]     = float[](0.5, 0.4, 0.3, 0.2);
    const float label_distances[3] = float[](0.8, 0.4, 0.1);

    for (int i = 0; i < 4; ++i) {
      float grid_fade = (1.0 - clamp(camDist / centerDist / distances[i], 0.0, 1.0));

      float latDeg        = (fsIn.lngLat.y / VP_PI) * 9000.0;
      float latDh         = mod(latDeg, spacings[i]);
      float latDhMirrored = (latDh > 0.5 * spacings[i]) ? spacings[i] - latDh : latDh;

      float wLatDeg = fwidth(latDeg);
      float latA    = clamp(abs(latDhMirrored / wLatDeg), 0.0, 1.0);

#if $SHOW_LAT_LONG
      fragColor.rgb = mix(fragColor.rgb, grid_color, (1.0 - latA) * grid_fade * intensities[i]);
#endif

      float lngDeg        = (fsIn.lngLat.x / VP_PI) * 9000.0;
      float lngDh         = mod(lngDeg, spacings[i]);
      float lngDhMirrored = (lngDh > 0.5 * spacings[i]) ? spacings[i] - lngDh : lngDh;

      float wLngDeg = fwidth(lngDeg);
      float lngA    = clamp(abs(lngDhMirrored / wLngDeg), 0.0, 1.0);

#if $SHOW_LAT_LONG
      fragColor.rgb = mix(fragColor.rgb, grid_color, (1.0 - lngA) * grid_fade * intensities[i]);
#endif

// print labels
#if $SHOW_LAT_LONG_LABELS
      if (i < 3) {
        float curr_label_fade =
            (1.0 - clamp(camDist / centerDist / label_distances[i] - 0.1, 0.0, 1.0));
        float next_label_fade =
            i < 2 ? (1.0 - clamp(camDist / centerDist / label_distances[i + 1] - 0.1, 0.0, 1.0))
                  : 0.0;

        // divide current cell in sub cells, 2 rows and 7 columns
        const int scaleX = 30;
        float     scaleY = 15 / (1 - abs(fsIn.lngLat.y / VP_PI));
        // float scaleY = 10 * abs(asin(idealNormal.y));

        vec2 rowCol = vec2(lngDh * scaleX, latDh * scaleY) / spacings[i];
        if (rowCol.x > scaleX - 7 && rowCol.x < scaleX && rowCol.y > 0 && rowCol.y < 2) {

          // compute mipmap level
          vec2 tcdx = dFdx(rowCol / scaleX);
          vec2 tcdy = dFdy(rowCol / scaleY);

          // get texture coordinate per digit and flip horizontally
          vec2 digitCoord = rowCol - ivec2(rowCol);

          float number, digit, suffix;

          if (rowCol.y < 1) {
            if (latDeg >= 0) {
              suffix = 10;
              number = 10 * (latDeg - latDh);
            } else {
              suffix = 11;
              number = -10 * (latDeg - latDh);
            }
          } else {
            if (lngDeg > 0) {
              suffix = 12;
              number = 10 * (lngDeg - lngDh + spacings[i]);
            } else {
              suffix = 13;
              number = -10 * (lngDeg - lngDh + spacings[i]);
            }
          }

          int numberInt = int(number / 50 + 0.5 * sign(number));

          switch (int(rowCol.x)) { // char column
          case scaleX - 1:
            digit = suffix;
            break;
          case scaleX - 2:
            digit = 15; // '°'
            break;
          case scaleX - 3:
            digit = mod(numberInt, 10);
            break;
          case scaleX - 4:
            digit = 14; // '.'
            break;
          case scaleX - 5:
            digit = mod((numberInt / 10), 10);
            break;
          case scaleX - 6:
            digit = numberInt >= 100 ? mod((numberInt / 100), 10) : -1;
            break;
          case scaleX - 7:
            digit = numberInt >= 1000 ? mod((numberInt / 1000), 10) : -1;
            break;
          default:
            digit = -1;
            break;
          }

          if (digit >= 0) {
            const float totalDigits = 16;
            digitCoord.x            = (digitCoord.x + digit) / totalDigits;

            float fade = 1.0 - textureGrad(fontTex, digitCoord, tcdx, tcdy).r;
            fade *= max(dot(viewDir / camDist, idealNormal), 0.0);
            fade *= clamp(curr_label_fade - next_label_fade, 0.0, 1.0);

            fragColor.rgb = mix(fragColor.rgb, grid_color, fade);
          }
        }
      }
#endif
    }
  }
#endif

#if $SHOW_HEIGHT_LINES
  {
    const float levels[4]      = float[](5000.0, 1000.0, 100.0, 10.0);
    const float distances[4]   = float[](1.5, 1.0, 0.5, 0.1);
    const float intensities[4] = float[](0.5, 0.4, 0.3, 0.2);

    float wh = fwidth(fsIn.height);

    for (int i = 0; i < 4; ++i) {
      float fade = (1.0 - clamp(camDist / centerDist / distances[i], 0.0, 1.0)) * intensities[i];

      // distance from IsoLevel
      float levelDh         = mod(fsIn.height, levels[i]);
      float levelDhMirrored = (levelDh > 0.5 * levels[i]) ? levels[i] - levelDh : levelDh;

      float level = clamp(abs(levelDhMirrored / wh), 0.0, 1.0);
      fragColor.rgb =
          mix(fragColor.rgb, (1.0 - intensities[i]) * fragColor.rgb, (1.0 - level) * fade);

      // layer shading
      float layer = clamp(abs((levels[i] - levelDh) / wh) * 0.1, 0.0, 1.0);
      fragColor.rgb =
          mix(fragColor.rgb, (1.0 - intensities[i]) * fragColor.rgb, (1.0 - layer) * fade * 0.5);
    }
  }
#endif
}
