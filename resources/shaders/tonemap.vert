////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

out vec2 vTexcoords;

void main() {
  vTexcoords  = vec2(gl_VertexID & 2, (gl_VertexID << 1) & 2);
  gl_Position = vec4(vTexcoords * 2.0 - 1.0, 0.0, 1.0);
}