////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

in vec2 vTexcoords;

layout(binding = 0) uniform usampler2D uImage;
layout(location = 0) out vec3 oColor;

uniform float uLuminanceMultiplicator;

void main() {
  vec2 temperatureLuminance = unpackHalf2x16(texture(uImage, vTexcoords).r);

  if (any(lessThanEqual(temperatureLuminance, vec2(0.0)))) {
    discard;
  }

  oColor = getStarColor(temperatureLuminance.x) * temperatureLuminance.y * uLuminanceMultiplicator;

#ifndef ENABLE_HDR
  oColor = Uncharted2Tonemap(oColor.rgb * 4e3);
#endif
}