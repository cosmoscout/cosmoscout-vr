////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// inputs
in vec3  iColor;
in float iMagnitude;
in vec2  iTexcoords;

// uniforms
uniform sampler2D iTexture;
uniform float     uSolidAngle;
uniform float     uLuminanceMultiplicator;
uniform float     uMinMagnitude;
uniform float     uMaxMagnitude;

// outputs
out vec4 oLuminance;

void main() {
  float dist      = min(1, length(iTexcoords));
  float luminance = magnitudeToLuminance(iMagnitude, uSolidAngle);

#ifdef DRAWMODE_DISC
  float fac = dist < 1 ? luminance : 0;
#endif

#ifdef DRAWMODE_SMOOTH_DISC
  // the brightness is basically a cone from above - to achieve the
  // same total brightness, we have to multiply it with three
  float fac = luminance * clamp(1 - dist, 0, 1) * 3;
#endif

#ifdef DRAWMODE_SCALED_DISC
  // The stars were scaled in this mode so we have to incorporate this here to
  // achieve the same total luminance.
  float scaleFac = mapRange(iMagnitude, uMinMagnitude, uMaxMagnitude, 3.0, 0.3);
  float fac      = luminance * clamp(1 - dist, 0, 1) * 3 / (scaleFac * scaleFac);
#endif

#ifdef DRAWMODE_GLARE_DISC
  float scaleFac = mapRange(sqrt(luminance), 0, 5, 1.0, 100.0);

  // In this mode, 20% of the brightness is drawn using a small smooth disc in the center
  // (just like DRAWMODE_SMOOTH_DISC) and the other 80% are drawn as glare using an inverse
  // quadratic falloff.

  // The billboard is scaled depending on the magnitude, but we want the disc to be the same
  // size for all stars. So we have to scale the distance accordingly.
  float disc = luminance * clamp(1 - dist * scaleFac, 0, 1) * 3;

  float falloff = max(0, 0.5 / pow(dist + 0.1, 2) - 0.5);
  float glare   = luminance * falloff / (scaleFac * scaleFac) * 0.5;

  float fac = 0.2 * glare + 0.8 * disc;
#endif

#ifdef DRAWMODE_SPRITE

  // The stars were scaled in this mode so we have to incorporate this here to achieve the
  // same total luminance.
  float scaleFac = mapRange(iMagnitude, uMinMagnitude, uMaxMagnitude, 10.0, 1.0);
  float fac      = texture(iTexture, iTexcoords * 0.5 + 0.5).r * luminance / (scaleFac * scaleFac);

  // A magic number here. This is the average brightness of the currently used
  // star texture (identify -format "%[fx:mean]\n" star.png).
  fac /= 0.0559036;
#endif

  vec3 vColor = iColor * fac * uLuminanceMultiplicator;

  oLuminance = vec4(vColor, 1.0);

#ifndef ENABLE_HDR
  // Random exposure adjustment to make the stars look good in non-HDR mode.
  oLuminance.rgb = Uncharted2Tonemap(oLuminance.rgb * uSolidAngle * 4e8);
#endif
}