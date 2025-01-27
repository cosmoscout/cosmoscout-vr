////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// inputs
in vec3 vView;

// uniforms
uniform sampler2D iTexture;
uniform vec4      cColor;

// outputs
layout(location = 0) out vec3 vOutColor;

float my_atan2(float a, float b) {
  return 2.0 * atan(a / (sqrt(b * b + a * a) + b));
}

void main() {
  const float PI       = 3.14159265359;
  vec3        view     = normalize(vView);
  vec2        texcoord = vec2(0.5 * my_atan2(view.x, -view.z) / PI, acos(view.y) / PI);
  vOutColor            = texture(iTexture, texcoord).rgb * cColor.rgb * cColor.a;
}