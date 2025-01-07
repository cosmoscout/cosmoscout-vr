////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

// inputs
in float vTemperature[];
in float vMagnitude[];

// uniforms
uniform mat4  uMatP;
uniform float uSolidAngle;
uniform float uMinMagnitude;
uniform float uMaxMagnitude;

// outputs
out vec3  iColor;
out float iMagnitude;
out vec2  iTexcoords;

void main() {
  iColor = getStarColor(vTemperature[0]);

  iMagnitude = vMagnitude[0];

  // Discard stars that are too bright or too dim.
  if (iMagnitude > uMaxMagnitude || iMagnitude < uMinMagnitude) {
    return;
  }

  // Discard stars that are outside the frustum. We only test the center position.
  vec4 center = uMatP * vec4(gl_in[0].gl_Position.xyz, 1);
  if (center.w <= 0) {
    return;
  }

  center /= center.w;
  if (center.x < -1 || center.x > 1 || center.y < -1 || center.y > 1) {
    return;
  }

  float dist = length(gl_in[0].gl_Position.xyz);
  vec3  y    = vec3(0, 1, 0);
  vec3  z    = gl_in[0].gl_Position.xyz / dist;
  vec3  x    = normalize(cross(z, y));
  y          = normalize(cross(z, x));

  const float offset[2] = float[2](0.5, -0.5);
  const float PI        = 3.14159265359;
  float       diameter  = sqrt(uSolidAngle / (4 * PI)) * 4.0;
  float       scale     = dist * diameter;

#ifdef DRAWMODE_SCALED_DISC
  // We scale the stars up a little in this mode so that they look more similar to
  // the other modes without the user having to move the star-size slider.
  scale *= mapRange(iMagnitude, uMinMagnitude, uMaxMagnitude, 3.0, 0.3);
#endif

#ifdef DRAWMODE_GLARE_DISC
  float luminance = magnitudeToLuminance(iMagnitude, uSolidAngle);
  scale *= mapRange(sqrt(luminance), 0, 5, 2.0, 100.0);
#endif

#ifdef DRAWMODE_SPRITE
  // We scale the stars up a little in this mode so that they look more similar to
  // the other modes without the user having to move the star-size slider.
  scale *= mapRange(iMagnitude, uMinMagnitude, uMaxMagnitude, 10.0, 1.0);
#endif

  for (int j = 0; j != 2; ++j) {
    for (int i = 0; i != 2; ++i) {
      iTexcoords = vec2(offset[i], offset[j]) * 2;
      vec3 pos   = gl_in[0].gl_Position.xyz + (offset[i] * x + offset[j] * y) * scale;

      gl_Position = uMatP * vec4(pos, 1);

      EmitVertex();
    }
  }
  EndPrimitive();
}