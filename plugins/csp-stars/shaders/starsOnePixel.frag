////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// inputs
in vec3  vColor;
in vec4  vScreenSpacePos;
in float vMagnitude;

// uniforms
uniform float uLuminanceMultiplicator;
uniform mat4  uInvP;
uniform vec2  uResolution;
uniform float uMinMagnitude;
uniform float uMaxMagnitude;
uniform float uSolidAngle;

// outputs
out vec4 oLuminance;

float getSolidAngle(vec3 a, vec3 b, vec3 c) {
  return 2 * atan(abs(dot(a, cross(b, c))) / (1 + dot(a, b) + dot(a, c) + dot(b, c)));
}

float getSolidAngleOfPixel(vec4 screenSpacePosition, vec2 resolution, mat4 invProjection) {
  vec2 pixel           = vec2(1.0) / resolution;
  vec4 pixelCorners[4] = vec4[4](screenSpacePosition + vec4(-pixel.x, -pixel.y, 0, 0),
      screenSpacePosition + vec4(+pixel.x, -pixel.y, 0, 0),
      screenSpacePosition + vec4(+pixel.x, +pixel.y, 0, 0),
      screenSpacePosition + vec4(-pixel.x, +pixel.y, 0, 0));

  for (int i = 0; i < 4; ++i) {
    pixelCorners[i]     = invProjection * pixelCorners[i];
    pixelCorners[i].xyz = normalize(pixelCorners[i].xyz);
  }

  return getSolidAngle(pixelCorners[0].xyz, pixelCorners[1].xyz, pixelCorners[2].xyz) +
         getSolidAngle(pixelCorners[0].xyz, pixelCorners[2].xyz, pixelCorners[3].xyz);
}

void main() {
  if (vMagnitude > uMaxMagnitude || vMagnitude < uMinMagnitude) {
    discard;
  }

  float solidAngle = getSolidAngleOfPixel(vScreenSpacePos, uResolution, uInvP);
  float luminance  = magnitudeToLuminance(vMagnitude, solidAngle);

  oLuminance = vec4(vColor * luminance * uLuminanceMultiplicator, 1.0);

#ifndef ENABLE_HDR
  oLuminance.rgb = Uncharted2Tonemap(oLuminance.rgb * uSolidAngle * 5e8);
#endif
}