////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

in vec2 vTexcoords;

layout(pixel_center_integer) in vec4 gl_FragCoord;

layout(binding = 0) uniform sampler2D uR;
layout(binding = 1) uniform sampler2D uG;
layout(binding = 2) uniform sampler2D uB;

layout(location = 0) out vec3 oColor;

void main() {
  vec3 color =
      vec3(texture(uR, vTexcoords).r, texture(uG, vTexcoords).r, texture(uB, vTexcoords).r);

  oColor = color;
}