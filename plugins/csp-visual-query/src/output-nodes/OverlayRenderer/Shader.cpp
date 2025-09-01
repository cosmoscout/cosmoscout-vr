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
  uniform vec2          uValueRange;
  uniform bool          uHasLUT;
  uniform int           uNumScalars;

  uniform dmat4         uMatInvMVP;

  uniform dvec2         uLonRange;
  uniform dvec2         uLatRange;
  uniform vec3          uRadii;

  uniform float         uAmbientBrightness;
  uniform float         uSunIlluminance;
  uniform vec3          uSunDirection;

  in vec2 texcoord;

  // From https://outerra.blogspot.com/2014/05/double-precision-approximations-for-map.html
  double atan2(double y, double x) {
    const double atan_tbl[] = {
    -3.333333333333333333333333333303396520128e-1LF,
     1.999999117496509842004185053319506031014e-1LF,
    -1.428514132711481940637283859690014415584e-1LF,
     1.110012236849539584126568416131750076191e-1LF,
    -8.993611617787817334566922323958104463948e-2LF,
     7.212338962134411520637759523226823838487e-2LF,
    -5.205055255952184339031830383744136009889e-2LF,
     2.938542391751121307313459297120064977888e-2LF,
    -1.079891788348568421355096111489189625479e-2LF,
     1.858552116405489677124095112269935093498e-3LF
    };

    /* argument reduction: 
       arctan (-x) = -arctan(x); 
       arctan (1/x) = 1/2 * pi - arctan (x), when x > 0
    */

    double ax = abs(x);
    double ay = abs(y);
    double t0 = max(ax, ay);
    double t1 = min(ax, ay);
    
    double a = 1 / t0;
    a *= t1;

    double s = a * a;
    double p = atan_tbl[9];

    p = fma( fma( fma( fma( fma( fma( fma( fma( fma( fma(p, s,
        atan_tbl[8]), s,
        atan_tbl[7]), s, 
        atan_tbl[6]), s,
        atan_tbl[5]), s,
        atan_tbl[4]), s,
        atan_tbl[3]), s,
        atan_tbl[2]), s,
        atan_tbl[1]), s,
        atan_tbl[0]), s*a, a);

    double r = ay > ax ? (1.57079632679489661923LF - p) : p;

    r = x < 0 ?  3.14159265358979323846LF - r : r;
    r = y < 0 ? -r : r;

    return r;
  }

  // From https://stackoverflow.com/questions/28969184/is-there-an-accurate-approximation-of-the-acos-function
  double asin_core (double a) {
    double q, r, s, t;

    s = a * a;
    q = s * s;
    r =             5.5579749017470502e-2;
    t =            -6.2027913464120114e-2;
    r = fma (r, q,  5.4224464349245036e-2);
    t = fma (t, q, -1.1326992890324464e-2);
    r = fma (r, q,  1.5268872539397656e-2);
    t = fma (t, q,  1.0493798473372081e-2);
    r = fma (r, q,  1.4106045900607047e-2);
    t = fma (t, q,  1.7339776384962050e-2);
    r = fma (r, q,  2.2372961589651054e-2);
    t = fma (t, q,  3.0381912707941005e-2);
    r = fma (r, q,  4.4642857881094775e-2);
    t = fma (t, q,  7.4999999991367292e-2);
    r = fma (r, s, t);
    r = fma (r, s,  1.6666666666670193e-1);
    t = a * s;
    r = fma (r, t, a);

    return r;
  }

  double acos(double a) {
    double r;

    r = (a > 0.0) ? -a : a; // avoid modifying the "sign" of NaNs
    if (r > -0.5625) {
        /* arccos(x) = pi/2 - arcsin(x) */
        r = fma (9.3282184640716537e-1, 1.6839188885261840e+0, asin_core (r));
    } else {
        /* arccos(x) = 2 * arcsin (sqrt ((1-x) / 2)) */
        r = 2.0 * asin_core (sqrt (fma (0.5, r, 0.5)));
    }
    if (!(a > 0.0) && (a >= -1.0)) { // avoid modifying the "sign" of NaNs
        /* arccos (-x) = pi - arccos(x) */
        r = fma (1.8656436928143307e+0, 1.6839188885261840e+0, -r);
    }
    return r;
  }

  double asin(double a) {
    return 1.57079632679489661923LF - acos(a);
  }

  dvec3 getPosition(float fDepth) {
    dvec4  posMS = uMatInvMVP * dvec4(2.0 * texcoord - 1.0, fDepth * 2.0 - 1.0 , 1.0);
    return posMS.xyz / posMS.w;
  }

  dvec3 surfaceToNormal(dvec3 cartesian, vec3 radii) {
    vec3 radii2        = radii * radii;
    vec3 oneOverRadii2 = 1.0 / radii2;
    return normalize(cartesian * dvec3(oneOverRadii2));
  }

  dvec2 surfaceToLngLat(dvec3 cartesian, vec3 radii) {
    dvec3 geodeticNormal = surfaceToNormal(cartesian, radii);
    return dvec2(atan2(geodeticNormal.x, geodeticNormal.z), asin(geodeticNormal.y));
  }

  void main() {
    vec2  vTexcoords = texcoord*textureSize(uDepthBuffer);
    float fDepth     = texture(uDepthBuffer, vTexcoords).r;

    if (fDepth == 1.0) {
      discard;
    } else {
      dvec3 worldPos = getPosition(fDepth);
      dvec2 lnglat    = surfaceToLngLat(worldPos, uRadii);

      if(lnglat.x > uLonRange.x && lnglat.x < uLonRange.y &&
         lnglat.y > uLatRange.x && lnglat.y < uLatRange.y) {

        double norm_u = (lnglat.x - uLonRange.x) / (uLonRange.y - uLonRange.x);
        double norm_v = (lnglat.y - uLatRange.x) / (uLatRange.y - uLatRange.x);
        vec2 newCoords = vec2(float(norm_u), float(1.0 - norm_v));

        FragColor = texture(uTexture, newCoords);

        if (uNumScalars < 4) {
          FragColor.a = 1.0;
        }

        if (uNumScalars == 1) {
          FragColor.rgb = FragColor.rrr;
        }

        FragColor.rgb = (FragColor.rgb - uValueRange.x) / (uValueRange.y - uValueRange.x);

        if (uHasLUT) {
          float value = max(FragColor.r, max(FragColor.g, FragColor.b));
          FragColor = texture(uLUT, value);
        }

      } else {
        discard;
      }
    }
  }
)";

} // namespace csp::visualquery