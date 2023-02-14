////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#version 330

// uniforms
uniform mat4 uMatInvMV;
uniform mat4 uMatInvP;

// outputs
out VaryingStruct {
  vec3 rayDir;
  vec3 rayOrigin;
  vec2 texcoords;
}
vsOut;

void main() {
  // Get observer position.
  vsOut.rayOrigin = uMatInvMV[3].xyz;

  // Get direction of the vertex / fragment.
  mat4 matInvMV = uMatInvMV;
  matInvMV[3]   = vec4(0, 0, 0, 1);
  vec2 position = vec2(gl_VertexID & 2, (gl_VertexID << 1) & 2) * 2.0 - 1.0;
  vsOut.rayDir  = (matInvMV * uMatInvP * vec4(position, 0, 1)).xyz;

  // For lookups in the depth and color buffers.
  vsOut.texcoords = position * 0.5 + 0.5;

  // No tranformation here since we draw a full screen quad.
  gl_Position = vec4(position, 0, 1);
}