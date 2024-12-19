////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

in vec2 vTexcoords;

layout(binding = 0) uniform usampler2D uImage;
layout(location = 0) out vec3 oColor;

void main() {

  uint value = texture(uImage, vTexcoords).r;

  // tEff is stored in the first 8 bits of the red channel
  float tEff = (value & 0xFF) * 100;

  // luminance is stored in the remaining 24 bits
  float luminance = (value >> 8) / pow(2.0, 20.0);

  if (tEff <= 0.0) {
    discard;
  }
  oColor = getStarColor(tEff) * luminance;

#ifndef ENABLE_HDR
  oColor = Uncharted2Tonemap(oColor.rgb * 1e3);
#endif
}