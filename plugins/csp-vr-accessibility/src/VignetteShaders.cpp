////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "FovVignette.hpp"

namespace csp::vraccessibility {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* FovVignette::VERT_SHADER = R"(
#version 330

// inputs
layout(location = 0) in vec2 iQuadPos;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;

void main() {
  vTexCoords  = vec2((iQuadPos.x + 1) / 2,
                     (iQuadPos.y + 1) / 2);
  vPosition   = vec3(iQuadPos.x, iQuadPos.y, -0.01);
  gl_Position = vec4(vPosition, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shader for static vignetting and circular vignette

const char* FovVignette::FRAG_SHADER_FADE = R"(
#version 330

uniform float uAspect;
uniform float uFade;
uniform vec4  uCustomColor;
uniform vec2  uRadii;
uniform bool  uDebug;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

void main() {
  if (uFade == 0 && !uDebug ) { discard; }

  float dist = sqrt(vPosition.x * vPosition.x + uAspect * uAspect * vPosition.y * vPosition.y);
  if (dist < uRadii[0] ) { discard; }

  float alpha = clamp((dist - uRadii[0]) / (uRadii[1] - uRadii[0]), 0, 1);

  oColor = uCustomColor;
  oColor.a *= alpha;

  if ( !uDebug ) { oColor.a *= uFade; }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shader for dynamic vignetting and circular vignette

const char* FovVignette::FRAG_SHADER_DYNRAD = R"(
#version 330

uniform float uAspect;
uniform vec4  uCustomColor;
uniform vec2  uRadii;
uniform bool  uDebug;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

void main() {
  float dist = sqrt(vPosition.x * vPosition.x + uAspect * uAspect * vPosition.y * vPosition.y);
  if (dist < uRadii[0] ) { discard; }

  float alpha = clamp((dist - uRadii[0]) / (uRadii[1] - uRadii[0]), 0, 1);

  oColor = uCustomColor;
  oColor.a *= alpha;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shader for static vignetting and vertical vignette

const char* FovVignette::FRAG_SHADER_FADE_VERTONLY = R"(
#version 330

uniform float uFade;
uniform vec4  uCustomColor;
uniform vec2  uRadii;
uniform bool  uDebug;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

void main() {
  if (uFade == 0 && !uDebug ) { discard; }

  float dist = 0;
  if (vPosition.y > 0) { 
    dist = vPosition.y; 
  } else {
    dist = vPosition.y * -0.7;
  }

  if (dist < uRadii[0] ) { discard; }
  
  float alpha = clamp((dist - uRadii[0]) / (uRadii[1] - uRadii[0]), 0, 1);

  oColor = uCustomColor;
  oColor.a *= alpha;

  if ( !uDebug ) { oColor.a *= uFade; }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shader for dynamic vignetting and vertical vignette

const char* FovVignette::FRAG_SHADER_DYNRAD_VERTONLY = R"(
#version 330

uniform vec4 uCustomColor;
uniform vec2 uRadii;
uniform bool uDebug;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

void main() {
  float dist = 0;
  if (vPosition.y > 0) {
    dist = vPosition.y;
  } else {
    dist = vPosition.y * -0.7;
  }

  if (dist < uRadii[0] ) { discard; }
  
  float alpha = clamp((dist - uRadii[0]) / (uRadii[1] - uRadii[0]), 0, 1);

  oColor = uCustomColor;
  oColor.a *= alpha;
}
)";

} // namespace csp::vraccessibility
