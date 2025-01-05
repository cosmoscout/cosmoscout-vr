////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

const float cParsecToMeter = 3.08567758e16;

// Returns one value from the spectral color table based on the given temperature.
// https://onlinelibrary.wiley.com/doi/10.1002/asna.202113868
vec3 getStarColor(float temperature) {
  const float tMin = 2300;
  const float tMax = 12000;

  const int  cSpectralColorsN    = 73;
  const vec3 cSpectralColors[73] = vec3[](vec3(1.0, 0.409, 0.078), vec3(1.0, 0.432, 0.093),
      vec3(1.0, 0.455, 0.109), vec3(1.0, 0.476, 0.126), vec3(1.0, 0.497, 0.144),
      vec3(1.0, 0.518, 0.163), vec3(1.0, 0.537, 0.182), vec3(1.0, 0.557, 0.202),
      vec3(1.0, 0.575, 0.223), vec3(1.0, 0.593, 0.244), vec3(1.0, 0.611, 0.266),
      vec3(1.0, 0.627, 0.289), vec3(1.0, 0.644, 0.311), vec3(1.0, 0.66, 0.335),
      vec3(1.0, 0.675, 0.358), vec3(1.0, 0.69, 0.382), vec3(1.0, 0.704, 0.405),
      vec3(1.0, 0.718, 0.429), vec3(1.0, 0.732, 0.454), vec3(1.0, 0.745, 0.478),
      vec3(1.0, 0.758, 0.502), vec3(1.0, 0.77, 0.527), vec3(1.0, 0.782, 0.551),
      vec3(1.0, 0.794, 0.575), vec3(1.0, 0.806, 0.599), vec3(1.0, 0.817, 0.624),
      vec3(1.0, 0.827, 0.648), vec3(1.0, 0.838, 0.672), vec3(1.0, 0.848, 0.696),
      vec3(1.0, 0.858, 0.719), vec3(1.0, 0.867, 0.743), vec3(1.0, 0.877, 0.766),
      vec3(1.0, 0.886, 0.789), vec3(1.0, 0.894, 0.812), vec3(1.0, 0.903, 0.835),
      vec3(1.0, 0.911, 0.858), vec3(1.0, 0.919, 0.88), vec3(1.0, 0.927, 0.902),
      vec3(1.0, 0.935, 0.924), vec3(1.0, 0.942, 0.946), vec3(1.0, 0.95, 0.967),
      vec3(1.0, 0.957, 0.989), vec3(0.991, 0.955, 1.0), vec3(0.971, 0.942, 1.0),
      vec3(0.952, 0.93, 1.0), vec3(0.934, 0.918, 1.0), vec3(0.917, 0.907, 1.0),
      vec3(0.901, 0.896, 1.0), vec3(0.87, 0.876, 1.0), vec3(0.843, 0.858, 1.0),
      vec3(0.817, 0.841, 1.0), vec3(0.794, 0.825, 1.0), vec3(0.773, 0.81, 1.0),
      vec3(0.753, 0.797, 1.0), vec3(0.735, 0.784, 1.0), vec3(0.718, 0.772, 1.0),
      vec3(0.703, 0.761, 1.0), vec3(0.688, 0.75, 1.0), vec3(0.674, 0.741, 1.0),
      vec3(0.662, 0.731, 1.0), vec3(0.65, 0.723, 1.0), vec3(0.639, 0.714, 1.0),
      vec3(0.628, 0.706, 1.0), vec3(0.618, 0.699, 1.0), vec3(0.609, 0.692, 1.0),
      vec3(0.6, 0.685, 1.0), vec3(0.592, 0.679, 1.0), vec3(0.584, 0.673, 1.0),
      vec3(0.577, 0.667, 1.0), vec3(0.57, 0.662, 1.0), vec3(0.563, 0.657, 1.0),
      vec3(0.557, 0.652, 1.0), vec3(0.55, 0.647, 1.0));

  float t   = clamp((temperature - tMin) / (tMax - tMin), 0.0, 1.0);
  int   idx = int(t * (cSpectralColorsN - 1));
  return cSpectralColors[idx];
}

// Returns the logarithm to the base 10.
float log10(float x) {
  return log(x) / log(10.0);
}

// Returns the apparent magnitude of a star based on its absolute magnitude and distance in parsecs.
float getApparentMagnitude(float absMagnitude, float distInParsce) {
  return absMagnitude + 5.0 * log10(distInParsce / 10.0);
}

// Formula from https://en.wikipedia.org/wiki/Surface_brightness.
float magnitudeToLuminance(float apparentMagnitude, float solidAngle) {
  const float steradiansToSquareArcSecs = 4.25e10;
  float surfaceBrightness = apparentMagnitude + 2.5 * log10(solidAngle * steradiansToSquareArcSecs);
  return 10.8e4 * pow(10, -0.4 * surfaceBrightness);
}

// Maps a value from one range to another.
float mapRange(float value, float oldMin, float oldMax, float newMin, float newMax) {
  return newMin + (clamp(value, oldMin, oldMax) - oldMin) * (newMax - newMin) / (oldMax - oldMin);
}

// Returns the solid angle of the triangle defined by the tips of the vectors a, b and c.
float getSolidAngle(vec3 a, vec3 b, vec3 c) {
  return 2 * atan(abs(dot(a, cross(b, c))) / (1 + dot(a, b) + dot(a, c) + dot(b, c)));
}

// Returns the solid angle of the pixel at screenSpacePosition.
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

// Returns the observer position in parsecs based on the inverse modelview matrix.
vec3 getObserverPosition(mat4 invMV) {
  return (invMV * vec4(0, 0, 0, 1) / cParsecToMeter).xyz;
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

vec3 SRGBtoLINEAR(vec3 srgbIn) {
  vec3 bLess = step(vec3(0.04045), srgbIn);
  return mix(srgbIn / vec3(12.92), pow((srgbIn + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
}
