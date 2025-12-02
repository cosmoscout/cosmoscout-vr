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
  vec3 observerPos = getObserverPosition(uInvMV);
  vMagnitude       = getApparentMagnitude(inAbsMagnitude, length(inPos - observerPos));
  vTemperature     = inTemperature;

  gl_Position = uMatMV * vec4(inPos * cParsecToMeter, 1);
}