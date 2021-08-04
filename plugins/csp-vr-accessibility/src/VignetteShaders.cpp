////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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

void main()
{
    vTexCoords  = vec2( (iQuadPos.x + 1) / 2,
                       (iQuadPos.y + 1) / 2 );
    vPosition   = vec3(iQuadPos.x, iQuadPos.y, -0.01);
    gl_Position = vec4(vPosition, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shader for static vignetting and circular vignette

const char* FovVignette::FRAG_SHADER_FADE = R"(
#version 330

uniform sampler2D uTexture;
uniform float uFade;
uniform vec4 uCustomColor;
uniform float uInnerRadius;
uniform float uOuterRadius;
uniform bool uDebug;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

void main()
{
    if (uFade == 0 && !uDebug ) { discard; }

    vec2 texSize = textureSize(uTexture, 0);
    float ratio = texSize.y / texSize.x;

    float dist = sqrt(vPosition.x * vPosition.x + ratio * ratio * vPosition.y * vPosition.y);
    if (dist < uInnerRadius ) { discard; }
    float r = ((dist - uInnerRadius) / (uOuterRadius - uInnerRadius));
    oColor = mix(texture(uTexture, vTexCoords), uCustomColor, r);
    if (dist > uOuterRadius) {
      oColor.rgb = uCustomColor.rgb;
    }

    if ( !uDebug ) { oColor.a = uFade; }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shader for dynamic vignetting and circular vignette

const char* FovVignette::FRAG_SHADER_DYNRAD = R"(
#version 330

uniform sampler2D uTexture;
uniform float uNormVelocity;
uniform vec4 uCustomColor;
uniform float uInnerRadius;
uniform float uOuterRadius;
uniform bool uDebug;
float radiusInner = uInnerRadius;
float radiusOuter = uOuterRadius;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

void main()
{
    if (uNormVelocity <= 0 && !uDebug ) { discard; }

    vec2 texSize = textureSize(uTexture, 0);
    float ratio = texSize.y / texSize.x;

    float dist = sqrt(vPosition.x * vPosition.x + ratio * ratio * vPosition.y * vPosition.y);

    if (dist < uInnerRadius ) { discard; }
    float r = ((dist - uInnerRadius) / (uOuterRadius - uInnerRadius));
    oColor = mix(texture(uTexture, vTexCoords), uCustomColor, r);
    if (dist > uOuterRadius) {
      oColor.rgb = uCustomColor.rgb;
    }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shader for static vignetting and vertical vignette

const char* FovVignette::FRAG_SHADER_FADE_VERTONLY = R"(
#version 330

uniform sampler2D uTexture;
uniform float uFade;
uniform vec4 uCustomColor;
uniform float uInnerRadius;
uniform float uOuterRadius;
uniform bool uDebug;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

void main()
{
    if (uFade == 0 && !uDebug ) { discard; }

    vec2 texSize = textureSize(uTexture, 0);
    float ratio = texSize.y / texSize.x;

    float dist = 0;
    if (vPosition.y > 0) { dist = vPosition.y; }
    else { dist = vPosition.y * -(0.7); }

    if (dist < uInnerRadius ) { discard; }
    float r = ((dist - uInnerRadius) / (uOuterRadius - uInnerRadius));
    oColor = mix(texture(uTexture, vTexCoords), uCustomColor, r);
    if (dist > uOuterRadius) {
      oColor.rgb = uCustomColor.rgb;
    }

    if ( !uDebug ) { oColor.a = uFade; }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shader for dynamic vignetting and vertical vignette

const char* FovVignette::FRAG_SHADER_DYNRAD_VERTONLY = R"(
#version 330

uniform sampler2D uTexture;
uniform float uNormVelocity;
uniform vec4 uCustomColor;
uniform float uInnerRadius;
uniform float uOuterRadius;
uniform bool uDebug;
float radiusInner = uInnerRadius;
float radiusOuter = uOuterRadius;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

void main()
{
    if (uNormVelocity <= 0 && !uDebug ) { discard; }

    vec2 texSize = textureSize(uTexture, 0);
    float ratio = texSize.y / texSize.x;

    float dist = 0;
    if (vPosition.y > 0) { dist = vPosition.y; }
    else { dist = vPosition.y * -(0.7); }

    if (dist < uInnerRadius ) { discard; }
    float r = ((dist - uInnerRadius) / (uOuterRadius - uInnerRadius));
    oColor = mix(texture(uTexture, vTexCoords), uCustomColor, r);
    if (dist > uOuterRadius) {
      oColor.rgb = uCustomColor.rgb;
    }
}
)";

} // namespace csp::vraccessibility