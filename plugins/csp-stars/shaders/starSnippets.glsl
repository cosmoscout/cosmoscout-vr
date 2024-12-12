////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

float log10(float x) {
  return log(x) / log(10.0);
}

float getApparentMagnitude(float absMagnitude, float distInParsce) {
  return absMagnitude + 5.0 * log10(distInParsce / 10.0);
}

// formula from https://en.wikipedia.org/wiki/Surface_brightness
float magnitudeToLuminance(float apparentMagnitude, float solidAngle) {
  const float steradiansToSquareArcSecs = 4.25e10;
  float surfaceBrightness = apparentMagnitude + 2.5 * log10(solidAngle * steradiansToSquareArcSecs);
  return 10.8e4 * pow(10, -0.4 * surfaceBrightness);
}

vec3 SRGBtoLINEAR(vec3 srgbIn) {
  vec3 bLess = step(vec3(0.04045), srgbIn);
  return mix(srgbIn / vec3(12.92), pow((srgbIn + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
}

float mapRange(float value, float oldMin, float oldMax, float newMin, float newMax) {
  return newMin + (clamp(value, oldMin, oldMax) - oldMin) * (newMax - newMin) / (oldMax - oldMin);
}

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