////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// inputs
layout(location = 0) in vec3 inPos;
layout(location = 1) in float inTemperature;
layout(location = 2) in float inAbsMagnitude;

// uniforms
uniform mat4 uMatMV;
uniform mat4 uInvMV;

// outputs
out float vTemperature;
out float vMagnitude;

void main() {
  const float parsecToMeter = 3.08567758e16;
  vec3        observerPos   = (uInvMV * vec4(0, 0, 0, 1) / parsecToMeter).xyz;

  vMagnitude   = getApparentMagnitude(inAbsMagnitude, length(inPos - observerPos));
  vTemperature = inTemperature;

  gl_Position = uMatMV * vec4(inPos * parsecToMeter, 1);
}