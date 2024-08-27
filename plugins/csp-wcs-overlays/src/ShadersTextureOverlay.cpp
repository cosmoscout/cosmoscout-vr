////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TextureOverlayRenderer.hpp"

#include <string>

namespace csp::wcsoverlays {

const std::string TextureOverlayRenderer::SURFACE_GEOM = R"(
  #version 330 core

  layout(points) in;
  layout(triangle_strip, max_vertices = 4) out;

  out vec2 texcoord;

  void main() {
    gl_Position = vec4( 1.0, 1.0, 0.5, 1.0 );
    texcoord = vec2( 1.0, 1.0 );
    EmitVertex();

    gl_Position = vec4(-1.0, 1.0, 0.5, 1.0 );
    texcoord = vec2( 0.0, 1.0 );
    EmitVertex();

    gl_Position = vec4( 1.0,-1.0, 0.5, 1.0 );
    texcoord = vec2( 1.0, 0.0 );
    EmitVertex();

    gl_Position = vec4(-1.0,-1.0, 0.5, 1.0 );
    texcoord = vec2( 0.0, 0.0 );
    EmitVertex();

    EndPrimitive();
  }
)";

const std::string TextureOverlayRenderer::SURFACE_VERT = R"(
  #version 330 core

  void main() {
  }
)";

const std::string TextureOverlayRenderer::SURFACE_FRAG = R"(
#version 440
out vec4 FragColor;

uniform sampler2DRect uDepthBuffer;
uniform sampler2D     uSimBuffer;

uniform sampler1D     uTransferFunction;

uniform mat4          uMatInvMVP;

uniform dvec2         uLonRange;
uniform dvec2         uLatRange;
uniform vec2          uRange;
uniform vec3          uRadii;

uniform vec3          uSunDirection;

uniform float         uDataMaxValue;

in vec2 texcoord;

const float PI = 3.14159265359;

vec3 GetPosition(float fDepth) {
  vec4  posMS = uMatInvMVP * vec4(2.0 * texcoord - 1.0, fDepth*2.0 - 1.0 , 1.0);
  return posMS.xyz / posMS.w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vec3 surfaceToNormal(vec3 cartesian, vec3 radii) {
    vec3 radii2        = radii * radii;
    vec3 oneOverRadii2 = 1.0 / radii2;
    return normalize(cartesian * oneOverRadii2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vec2 surfaceToLngLat(vec3 cartesian, vec3 radii) {
    vec3 geodeticNormal = surfaceToNormal(cartesian, radii);
    return vec2(atan(geodeticNormal.x, geodeticNormal.z), asin(geodeticNormal.y));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void main() {
  vec2  vTexcoords = texcoord*textureSize(uDepthBuffer);
  float fDepth     = texture(uDepthBuffer, vTexcoords).r;
  if (fDepth == 1.0) {
    discard;
  } else {
    vec3 worldPos  = GetPosition(fDepth);
    vec2 lnglat    = surfaceToLngLat(vec3(worldPos.x, worldPos.y, worldPos.z), uRadii);

    FragColor = vec4(worldPos, 1.0);

    if (lnglat.x > uLonRange.x && lnglat.x < uLonRange.y &&
        lnglat.y > uLatRange.x && lnglat.y < uLatRange.y) {
      double norm_u = (lnglat.x - uLonRange.x) / (uLonRange.y - uLonRange.x);
      double norm_v = (lnglat.y - uLatRange.x) / (uLatRange.y - uLatRange.x);
      vec2 newCoords = vec2(float(norm_u), float(1.0 - norm_v));

      float value = texture(uSimBuffer, newCoords).r;
      if (value < 0) {
        discard;
      }

      value = value * uDataMaxValue;

      // Texture lookup and color mapping
      float normSimValue  = (value - uRange.x) / (uRange.y - uRange.x);
      vec4 color = texture(uTransferFunction, normSimValue);

      vec3 result = color.rgb;
      #ifdef ENABLE_LIGHTING
        // Lighting using a normal calculated from partial derivative
        vec3 dx = dFdx( worldPos );
        vec3 dy = dFdy( worldPos );

        vec3 N = normalize(cross(dx, dy));
        float NdotL = dot(N, -uSunDirection);

        float ambientStrength = 0.2;
        vec3 lightColor = vec3(1.0, 1.0, 1.0);
        vec3 ambient = ambientStrength * lightColor;
        vec3 diffuse = lightColor * NdotL;
        result = color.rgb;
      #endif

      FragColor = vec4(result, color.a);
    } else {
      discard;
    }
  }
}
)";

} // namespace csp::wcsoverlays