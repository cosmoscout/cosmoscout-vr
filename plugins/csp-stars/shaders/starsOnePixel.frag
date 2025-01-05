////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// inputs
in float vTemperature;
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

void main() {
  if (vMagnitude > uMaxMagnitude || vMagnitude < uMinMagnitude) {
    discard;
  }

  float solidAngle = getSolidAngleOfPixel(vScreenSpacePos, uResolution, uInvP);
  float luminance  = magnitudeToLuminance(vMagnitude, solidAngle);

  oLuminance = vec4(getStarColor(vTemperature) * luminance * uLuminanceMultiplicator, 1.0);

#ifndef ENABLE_HDR
  // Random exposure adjustment to make the stars look good in non-HDR mode.
  oLuminance.rgb = Uncharted2Tonemap(oLuminance.rgb * uSolidAngle * 4e8);
#endif
}