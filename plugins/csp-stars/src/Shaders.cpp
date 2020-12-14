////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Stars.hpp"

namespace csp::stars {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Stars::cStarsSnippets = R"(
float log10(float x) {
    return log(x) / log(10.0);
}

float getApparentMagnitude(float absMagnitude, float distInParsce) {
    return absMagnitude + 5.0*log10(distInParsce / 10.0);
}

// formula from https://en.wikipedia.org/wiki/Surface_brightness
float magnitudeToLuminance(float apparentMagnitude, float solidAngle) {
    const float steradiansToSquareArcSecs = 4.25e10;
    float surfaceBrightness = apparentMagnitude + 2.5 * log10(solidAngle * steradiansToSquareArcSecs);
    return 10.8e4 * pow(10, -0.4 * surfaceBrightness);
}

vec3 SRGBtoLINEAR(vec3 srgbIn) {
  vec3 bLess = step(vec3(0.04045),srgbIn);
  return mix( srgbIn/vec3(12.92), pow((srgbIn+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
}

// http://filmicworlds.com/blog/filmic-tonemapping-operators/
float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 Uncharted2Tonemap(vec3 x) {
  return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Stars::cStarsVert = R"(
// inputs
layout(location = 0) in vec2  inDir;
layout(location = 1) in float inDist;
layout(location = 2) in vec3  inColor;
layout(location = 3) in float inAbsMagnitude;
                                                                            
// uniforms
uniform mat4 uMatMV;
uniform mat4 uInvMV;

// outputs
out vec3  vColor;
out float vMagnitude;

void main() {
    vec3 starPos = vec3(
        cos(inDir.x) * cos(inDir.y) * inDist,
        sin(inDir.x) * inDist,
        cos(inDir.x) * sin(inDir.y) * inDist);

    const float parsecToMeter = 3.08567758e16;
    vec3 observerPos = (uInvMV * vec4(0, 0, 0, 1) / parsecToMeter).xyz;

    vMagnitude = getApparentMagnitude(inAbsMagnitude, length(starPos-observerPos));
    vColor = inColor;

    gl_Position = uMatMV * vec4(starPos*parsecToMeter, 1);
}

)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Stars::cStarsGeom = R"(
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

// inputs
in vec3  vColor[];
in float vMagnitude[];

// uniforms
uniform mat4  uMatP;
uniform float uSolidAngle;
uniform float uMinMagnitude;
uniform float uMaxMagnitude;

// outputs
out vec3  iColor;
out float iMagnitude;
out vec2  iTexcoords;

void main() {
    iColor = SRGBtoLINEAR(vColor[0]);

    iMagnitude = vMagnitude[0];

    if (iMagnitude > uMaxMagnitude || iMagnitude < uMinMagnitude) {
        return;
    }

    float dist = length(gl_in[0].gl_Position.xyz);
    vec3 y = vec3(0, 1, 0);
    vec3 z = gl_in[0].gl_Position.xyz / dist;
    vec3 x = normalize(cross(z, y));
    y = normalize(cross(z, x));

    const float yo[2] = float[2](0.5, -0.5);
    const float xo[2] = float[2](0.5, -0.5);

    const float PI = 3.14159265359;
    float diameter = 2 * sqrt(1 - pow(1-uSolidAngle/(2*PI), 2.0));
    float scale = dist * diameter;

    #ifdef DRAWMODE_SCALED_DISC
        scale *= 1.0 - (iMagnitude - uMinMagnitude) / (uMaxMagnitude - uMinMagnitude);

        // We scale the stars up a little in this mode so that they look more similar to
        // the other modes without the user having to move the star-size slider.
        scale *= 3.0;
    #endif

    #ifdef DRAWMODE_SPRITE
        scale *= 1.0 - (iMagnitude - uMinMagnitude) / (uMaxMagnitude - uMinMagnitude);

        // We scale the stars up a little in this mode so that they look more similar to
        // the other modes without the user having to move the star-size slider.
        scale *= 10.0;
    #endif

    for(int j=0; j!=2; ++j) {
        for(int i=0; i!=2; ++i) {
            iTexcoords = vec2(xo[i], yo[j])*2;

            vec3 pos = gl_in[0].gl_Position.xyz + (xo[i] * x + yo[j] * y) * scale;

            gl_Position = uMatP * vec4(pos, 1);

            if (gl_Position.w > 0) {
                gl_Position /= gl_Position.w;
                if (gl_Position.z >= 1) {
                    gl_Position.z = 0.999999;
                }
                EmitVertex();
            }
        }
    }
    EndPrimitive();
}

)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Stars::cStarsFrag = R"(
// inputs
in vec3  iColor;
in float iMagnitude;
in vec2  iTexcoords;

// uniforms
uniform sampler2D iTexture;
uniform float uSolidAngle;
uniform float uLuminanceMultiplicator;
uniform float uMinMagnitude;
uniform float uMaxMagnitude;

// outputs
out vec4 oLuminance;

void main() {
    float dist = min(1, length(iTexcoords));
    float luminance = magnitudeToLuminance(iMagnitude, uSolidAngle);

    #ifdef DRAWMODE_DISC
        float fac = dist < 1 ? luminance : 0;
    #endif

    #ifdef DRAWMODE_SMOOTH_DISC
        // the brightness is basically a cone from above - to achieve the
        // same total brightness, we have to multiply it with three
        float fac = luminance * clamp(1-dist, 0, 1) * 3;
    #endif
    
    #ifdef DRAWMODE_SCALED_DISC
        float scaleFac = 1.0 - (iMagnitude - uMinMagnitude) / (uMaxMagnitude - uMinMagnitude);

        // The stars were scaled up a little in this mode so we have to incorporate this
        // here to achieve the same total luminance.
        scaleFac *= 3.0;
        float fac = luminance * clamp(1-dist, 0, 1) * 3 / (scaleFac * scaleFac);
    #endif

    #ifdef DRAWMODE_SPRITE
        float scaleFac = 1.0 - (iMagnitude - uMinMagnitude) / (uMaxMagnitude - uMinMagnitude);

        // The stars were scaled up a little in this mode so we have to incorporate this
        // here to achieve the same total luminance.
        scaleFac *= 10.0;
        float fac = texture(iTexture, iTexcoords * 0.5 + 0.5).r * luminance / (scaleFac * scaleFac);

        // A magic number here. This is the average brightness of the currently used
        // star texture (identify -format "%[fx:mean]\n" star.png).
        fac /= 0.0559036;
    #endif

    vec3 vColor = iColor * fac * uLuminanceMultiplicator;

    oLuminance  = vec4(vColor, 1.0);

    #ifndef ENABLE_HDR
        oLuminance.rgb = Uncharted2Tonemap(oLuminance.rgb * uSolidAngle * 5e8);
    #endif
}

)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Stars::cStarsVertOnePixel = R"(

// inputs
layout(location = 0) in vec2  inDir;
layout(location = 1) in float inDist;
layout(location = 2) in vec3  inColor;
layout(location = 3) in float inAbsMagnitude;
                                                                            
// uniforms
uniform mat4 uMatMV;
uniform mat4 uMatP;
uniform mat4 uInvMV;

// outputs
out vec3  vColor;
out vec4  vScreenSpacePos;
out float vMagnitude;

void main() {
    vec3 starPos = vec3(
        cos(inDir.x) * cos(inDir.y) * inDist,
        sin(inDir.x) * inDist,
        cos(inDir.x) * sin(inDir.y) * inDist);

    const float parsecToMeter = 3.08567758e16;
    vec3 observerPos = (uInvMV * vec4(0, 0, 0, 1) / parsecToMeter).xyz;

    vMagnitude = getApparentMagnitude(inAbsMagnitude, length(starPos-observerPos));
    
    vColor = SRGBtoLINEAR(inColor);

    vScreenSpacePos = uMatP * uMatMV * vec4(starPos*parsecToMeter, 1);
    
    if (vScreenSpacePos.w > 0) {
        vScreenSpacePos /= vScreenSpacePos.w;
        if (vScreenSpacePos.z >= 1) {
            vScreenSpacePos.z = 0.999999;
        }
    }

    gl_Position = vScreenSpacePos;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Stars::cStarsFragOnePixel = R"(

// inputs
in vec3  vColor;
in vec4  vScreenSpacePos;
in float vMagnitude;

// uniforms
uniform float uLuminanceMultiplicator;
uniform mat4 uInvP;
uniform vec2 uResolution;
uniform float uMinMagnitude;
uniform float uMaxMagnitude;
uniform float uSolidAngle;

// outputs
out vec4 oLuminance;

float getSolidAngle(vec3 a, vec3 b, vec3 c) {
    return 2 * atan(abs(dot(a, cross(b, c))) / (1 + dot(a, b) + dot(a, c) + dot(b, c)));
}

float getSolidAngleOfPixel(vec4 screenSpacePosition, vec2 resolution, mat4 invProjection) {
    vec2 pixel = vec2(1.0) / resolution;
    vec4 pixelCorners[4] = vec4[4](
        screenSpacePosition + vec4(- pixel.x, - pixel.y, 0, 0),
        screenSpacePosition + vec4(+ pixel.x, - pixel.y, 0, 0),
        screenSpacePosition + vec4(+ pixel.x, + pixel.y, 0, 0),
        screenSpacePosition + vec4(- pixel.x, + pixel.y, 0, 0)
    );

    for (int i=0; i<4; ++i) {
        pixelCorners[i] = invProjection * pixelCorners[i];
        pixelCorners[i].xyz = normalize(pixelCorners[i].xyz);
    }

    return getSolidAngle(pixelCorners[0].xyz, pixelCorners[1].xyz, pixelCorners[2].xyz)
         + getSolidAngle(pixelCorners[0].xyz, pixelCorners[2].xyz, pixelCorners[3].xyz);
}

void main() {
    if (vMagnitude > uMaxMagnitude || vMagnitude < uMinMagnitude) {
        discard;
    }

    float solidAngle = getSolidAngleOfPixel(vScreenSpacePos, uResolution, uInvP);
    float luminance = magnitudeToLuminance(vMagnitude, solidAngle);

    oLuminance = vec4(vColor * luminance * uLuminanceMultiplicator, 1.0);

    #ifndef ENABLE_HDR
        oLuminance.rgb = Uncharted2Tonemap(oLuminance.rgb * uSolidAngle * 5e8);
    #endif
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Stars::cBackgroundVert = R"(
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
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Stars::cBackgroundFrag = R"(
// inputs
in vec3 vView;

// uniforms
uniform sampler2D iTexture;
uniform vec4      cColor;

// outputs
layout(location = 0) out vec3 vOutColor;

float my_atan2(float a, float b) {
    return 2.0 * atan(a/(sqrt(b*b + a*a) + b));
}

void main() {
    const float PI = 3.14159265359;
    vec3 view = normalize(vView);
    vec2 texcoord = vec2(0.5*my_atan2(view.x, -view.z)/PI, acos(view.y)/PI);
    vOutColor = texture(iTexture, texcoord).rgb * cColor.rgb * cColor.a;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::stars
