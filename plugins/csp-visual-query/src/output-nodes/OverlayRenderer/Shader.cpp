////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Renderer.hpp"

#include <string>

namespace csp::visualquery {

const std::string Renderer::SURFACE_GEOM = R"(
  #version 330 core

  layout(points) in;
  layout(triangle_strip, max_vertices = 4) out;

  out vec2 texcoord;

  void main() {
    gl_Position = vec4(1.0, 1.0, 0.5, 1.0);
    texcoord = vec2(1.0, 1.0);
    EmitVertex();

    gl_Position = vec4(-1.0, 1.0, 0.5, 1.0);
    texcoord = vec2(0.0, 1.0);
    EmitVertex();

    gl_Position = vec4(1.0,-1.0, 0.5, 1.0);
    texcoord = vec2(1.0, 0.0);
    EmitVertex();

    gl_Position = vec4(-1.0,-1.0, 0.5, 1.0);
    texcoord = vec2(0.0, 0.0);
    EmitVertex();

    EndPrimitive();
  }
)";

const std::string Renderer::SURFACE_VERT = R"(
  #version 330 core

  void main() {}
)";

const std::string Renderer::SURFACE_FRAG = R"(
  #version 440

  out vec4 FragColor;

  uniform sampler2DRect uDepthBuffer;
  uniform sampler2D     uTexture;
  uniform sampler1D     uLUT;
  uniform vec2          uLUTRange;

  uniform dmat4         uMatInvMVP;

  uniform dvec2         uLonRange;
  uniform dvec2         uLatRange;
  uniform vec3          uRadii;

  uniform float         uAmbientBrightness;
  uniform float         uSunIlluminance;
  uniform vec3          uSunDirection;

  in vec2 texcoord;

  const float PI = 3.14159265359;

  dvec3 getPosition(float fDepth) {
    dvec4  posMS = uMatInvMVP * dvec4(2.0 * texcoord - 1.0, fDepth * 2.0 - 1.0 , 1.0);
    return posMS.xyz / posMS.w;
  }

  vec3 surfaceToNormal(vec3 cartesian, vec3 radii) {
    vec3 radii2        = radii * radii;
    vec3 oneOverRadii2 = 1.0 / radii2;
    return normalize(cartesian * oneOverRadii2);
  }

  vec2 surfaceToLngLat(vec3 cartesian, vec3 radii) {
    vec3 geodeticNormal = surfaceToNormal(cartesian, radii);
    return vec2(atan(geodeticNormal.x, geodeticNormal.z), asin(geodeticNormal.y));
  }

  void main() {
    vec2  vTexcoords = texcoord*textureSize(uDepthBuffer);
    float fDepth     = texture(uDepthBuffer, vTexcoords).r;

    if (fDepth == 1.0) {
      discard;
    } else {
      dvec3 worldPos = getPosition(fDepth);
      vec2 lnglat    = surfaceToLngLat(vec3(worldPos), uRadii);

      if(lnglat.x > uLonRange.x && lnglat.x < uLonRange.y &&
         lnglat.y > uLatRange.x && lnglat.y < uLatRange.y) {

        double norm_u = (lnglat.x - uLonRange.x) / (uLonRange.y - uLonRange.x);
        double norm_v = (lnglat.y - uLatRange.x) / (uLatRange.y - uLatRange.x);
        vec2 newCoords = vec2(float(norm_u), float(1.0 - norm_v));

        float value = texture(uTexture, newCoords).r;
        // value = clamp((value - uLUTRange.x) / (uLUTRange.y - uLUTRange.x), 0.0, 1.0);
        FragColor = texture(uLUT, value);
      } else {
        discard;
      }
    }
  }
)";

} // namespace csp::visualquery