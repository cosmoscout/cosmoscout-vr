////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// inputs
layout(location = 0) in vec2 vPosition;

// uniforms
uniform mat4 uInvMVP;
uniform mat4 uInvMV;

// outputs
out vec3 vView;

void main() {
  vec3 vRayOrigin = (uInvMV * vec4(0, 0, 0, 1)).xyz;
  vec4 vRayEnd    = uInvMVP * vec4(vPosition, 0, 1);
  vView           = vRayEnd.xyz / vRayEnd.w - vRayOrigin;
  gl_Position     = vec4(vPosition, 1, 1);
}