#ifndef CS_ECLIPSE_SHADOWS_GLSL
#define CS_ECLIPSE_SHADOWS_GLSL

const float ECLIPSE_TEX_SHADOW_EXPONENT = 1.0;
const int   ECLIPSE_MAX_BODIES          = 8;
const float ECLIPSE_PI                  = 3.14159265358979323846;
const float ECLIPSE_TWO_PI              = 2.0 * ECLIPSE_PI;

uniform int       uEclipseMode;
uniform vec4      uEclipseSun;
uniform int       uEclipseNumOccluders;
uniform vec4      uEclipseOccluders[ECLIPSE_MAX_BODIES];
uniform sampler2D uEclipseShadowMaps[ECLIPSE_MAX_BODIES];

// Returns the surface area of a circle.
float _eclipseGetCircleArea(float r) {
  return ECLIPSE_PI * r * r;
}

// Returns the surface area of a spherical cap on a unit sphere.
float _eclipseGetCapArea(float r) {
  return 2.0 * ECLIPSE_PI * (1.0 - cos(r));
}

float _eclipseGetCircleIntersection(float radiusA, float radiusB, float centerDistance) {

  // No intersection
  if (centerDistance >= radiusA + radiusB) {
    return 0.0;
  }

  // One circle fully in the other (total eclipse)
  if (min(radiusA, radiusB) <= max(radiusA, radiusB) - centerDistance) {
    return _eclipseGetCircleArea(min(radiusA, radiusB));
  }

  float d = centerDistance;

  float rrA = radiusA * radiusA;
  float rrB = radiusB * radiusB;
  float dd  = d * d;

  float d1 = fma(radiusA, radiusA, fma(-radiusB, radiusB, dd)) / (2 * d);
  float d2 = d - d1;

  float fourth = -d2 * sqrt(fma(-d2, d2, rrB));
  float third  = fma(rrB, acos(d2 / radiusB), fourth);
  float second = fma(-d1, sqrt(fma(-d1, d1, rrA)), third);

  return fma(rrA, acos(d1 / radiusA), second);
}

// Returns the intersection area of two spherical caps with radii radiusA and radiusB whose center
// points are centerDistance away from each other. All values are given as angles on the unit
// sphere.
float _eclipseGetCapIntersection(float radiusA, float radiusB, float centerDistance) {

  // No intersection
  if (centerDistance >= radiusA + radiusB) {
    return 0.0;
  }

  // One circle fully in the other
  if (min(radiusA, radiusB) <= max(radiusA, radiusB) - centerDistance) {
    return _eclipseGetCapArea(min(radiusA, radiusB));
  }

  float sinD  = sin(centerDistance);
  float cosD  = cos(centerDistance);
  float sinRA = sin(radiusA);
  float sinRB = sin(radiusB);
  float cosRA = cos(radiusA);
  float cosRB = cos(radiusB);

  return 2.0 * (ECLIPSE_PI - acos(cosD / (sinRA * sinRB) - (cosRA * cosRB) / (sinRA * sinRB)) -
                   acos(cosRB / (sinD * sinRA) - (cosD * cosRA) / (sinD * sinRA)) * cosRA -
                   acos(cosRA / (sinD * sinRB) - (cosD * cosRB) / (sinD * sinRB)) * cosRB);
}

float _eclipseGetCapIntersectionApprox(float radiusA, float radiusB, float centerDistance) {

  // No intersection
  if (centerDistance >= radiusA + radiusB) {
    return 0.0;
  }

  // One circle fully in the other
  if (min(radiusA, radiusB) <= max(radiusA, radiusB) - centerDistance) {
    return _eclipseGetCapArea(min(radiusA, radiusB));
  }

  float diff   = abs(radiusA - radiusB);
  float interp = smoothstep(
      0.0, 1.0, 1.0 - clamp((centerDistance - diff) / (radiusA + radiusB - diff), 0.0, 1.0));

  return interp * _eclipseGetCapArea(min(radiusA, radiusB));
}

// This returns basically acos(dot(v1, v2)), but seems to have less floating point errors.
float _eclipseGetAngle(vec3 v1, vec3 v2) {
  float c = dot(v1 - v2, v1 - v2);
  return 2.0 * atan(sqrt(c), sqrt(4 - c));
}

vec3 _eclipseProjectPointOnRay(vec3 origin, vec3 direction, vec3 p) {
  vec3 ap = p - origin;
  vec3 ab = direction - origin;
  return origin + (dot(ap, ab) / dot(ab, ab)) * ab;
}

vec4 _eclipseGetBodyDirAngle(vec4 body, vec3 position) {
  vec3  bodyPos   = body.xyz - position;
  float bodyDist  = length(bodyPos);
  vec3  bodyDir   = bodyPos / bodyDist;
  float bodyAngle = asin(body.w / bodyDist);

  return vec4(bodyDir, bodyAngle);
}

vec3 _eclipseShadowMapLookup(int i, vec3 position) {
  float sunDistance = length(uEclipseSun.xyz - uEclipseOccluders[i].xyz);
  float sunRadius   = uEclipseSun.w;

  float rOcc   = uEclipseOccluders[i].w;
  float dOcc   = sunDistance / (sunRadius / rOcc + 1);
  float y0     = rOcc / dOcc * sqrt(dOcc * dOcc - rOcc * rOcc);
  float xOcc   = sqrt(rOcc * rOcc - y0 * y0);
  float xUmbra = (sunDistance * rOcc) / (sunRadius - rOcc);
  float xF     = xOcc - dOcc;
  float fac    = y0 / -xF;

  vec3 pSunOcc = _eclipseProjectPointOnRay(
      uEclipseOccluders[i].xyz, uEclipseOccluders[i].xyz - uEclipseSun.xyz, position);
  vec2 pos = vec2(distance(pSunOcc, uEclipseOccluders[i].xyz), distance(pSunOcc, position));

  // Todo: Maybe pos.x += xOcc is required?

  float alphaX = pos.x / (pos.x + xUmbra);
  float alphaY = pos.y / (fac * (pos.x - xF));

  float x = pow(alphaX, 1.0 / ECLIPSE_TEX_SHADOW_EXPONENT);
  float y = alphaY;

  if (x <= 0.0 || x >= 1.0 || y <= 0.0 || y >= 1.0) {
    return vec3(1.0);
  }

  return texture(uEclipseShadowMaps[i], vec2(x, 1 - y)).rgb;
}

// https://github.com/OpenSpace/OpenSpace/blob/master/modules/globebrowsing/shaders/renderer_fs.glsl#L115
vec3 _eclipseGetOpenSpaceParams(int i, vec3 position) {
  float sunDistance = length(uEclipseSun.xyz - uEclipseOccluders[i].xyz);
  float sunRadius   = uEclipseSun.w;
  float rOcc        = uEclipseOccluders[i].w;

  vec3 pc      = uEclipseOccluders[i].xyz - position;
  vec3 sc_norm = (uEclipseOccluders[i].xyz - uEclipseSun.xyz) / sunDistance;
  vec3 pc_proj = dot(pc, sc_norm) * sc_norm;
  vec3 d       = pc - pc_proj;

  float length_d       = length(d);
  float length_pc_proj = length(pc_proj);

  float xp = sunDistance / (sunRadius + rOcc);
  float xu = rOcc * sunDistance / (sunRadius - rOcc);

  float r_p_pi = rOcc * (length_pc_proj + xp) / xp;
  float r_u_pi = rOcc * (xu - length_pc_proj) / xu;

  return vec3(length_d, r_u_pi, r_p_pi);
}

vec3 _eclipseOpenSpace(int i, vec3 position) {
  vec3 params = _eclipseGetOpenSpaceParams(i, position);

  if (params.x < params.y) { // umbra
    // return vec3(0.0);
    return vec3(sqrt(params.y / (params.y + pow(params.x, 2.0))));
  } else if (params.x < params.z) { // penumbra
    // return vec3((params.x - params.y) / params.z);
    return vec3(params.x / params.z);
  }

  return vec3(1.0);
}

// https://github.com/CelestiaProject/Celestia/blob/master/src/celengine/shadermanager.cpp#L3811
// https://github.com/CelestiaProject/Celestia/blob/master/src/celengine/render.cpp#L2969
// https://github.com/CelestiaProject/Celestia/blob/master/src/celengine/shadermanager.cpp#L1344
vec3 _eclipseCelestia(int i, vec3 position) {
  // float distToSun    = length(uEclipseSun.xyz - position);
  // float appSunRadius = uEclipseSun.w / distToSun;

  // float distToCaster      = length(uEclipseOccluders[i].xyz - position);
  // float appOccluderRadius = uEclipseOccluders[i].w / distToCaster;

  // float penumbraRadius = (1 + appSunRadius / appOccluderRadius) * uEclipseOccluders[i].w;

  // float umbraRadius =
  //     uEclipseOccluders[i].w * (appOccluderRadius - appSunRadius) / appOccluderRadius;
  // float maxDepth = min(1.0, pow(appOccluderRadius / appSunRadius, 2.0));

  // float umbra   = umbraRadius / penumbraRadius;
  // float falloff = -maxDepth / max(0.001, 1.0 - abs(umbra));

  // vec3 params = _eclipseGetOpenSpaceParams(i, position);
  // float shadowR = clamp((2.0 * params.x - 1.0) * falloff, 0.0, maxDepth);

  // return vec3(1.0 - shadowR);

  vec3 params = _eclipseGetOpenSpaceParams(i, position);

  if (params.x < params.y) { // umbra
    return vec3(0.0);
  } else if (params.x < params.z) { // penumbra
    return vec3((params.x - params.y) / (params.z - params.y));
  }

  return vec3(1.0);
}

vec3 getEclipseShadow(vec3 position) {

  // None.
  if (uEclipseMode == 0) {
    return vec3(1.0);
  }

  // Debug.
  if (uEclipseMode == 1) {
    vec3 light = vec3(1.0);

    vec4 sunDirAngle = _eclipseGetBodyDirAngle(uEclipseSun, position);

    for (int i = 0; i < uEclipseNumOccluders; ++i) {

      vec4  bodyDirAngle = _eclipseGetBodyDirAngle(uEclipseOccluders[i], position);
      float sunBodyDist  = _eclipseGetAngle(sunDirAngle.xyz, bodyDirAngle.xyz);

      if (sunDirAngle.w < bodyDirAngle.w - sunBodyDist) { // Total eclipse.
        light *= vec3(0.25);
      } else if (sunBodyDist < sunDirAngle.w + bodyDirAngle.w) { // Partial eclipse.
        light *= vec3(0.75);
      }
    }

    return light;
  }

  // Celestia.
  if (uEclipseMode == 2) {
    vec3 light = vec3(1.0);
    for (int i = 0; i < uEclipseNumOccluders; ++i) {
      light *= _eclipseCelestia(i, position);
    }

    return light;
  }

  // Cosmographia.
  if (uEclipseMode == 3) {
    vec3 light = vec3(1.0);
    for (int i = 0; i < uEclipseNumOccluders; ++i) {
      light *= _eclipseCelestia(i, position);
    }

    return light;
  }

  // OpenSpace.
  if (uEclipseMode == 4) {
    vec3 light = vec3(1.0);
    for (int i = 0; i < uEclipseNumOccluders; ++i) {
      light *= _eclipseOpenSpace(i, position);
    }

    return light;
  }

  // Circle Intersection, Approximated Spherical Cap Intersection, or Spherical Cap Intersection.
  if (uEclipseMode == 5 || uEclipseMode == 6 || uEclipseMode == 7) {

    vec3 light = vec3(1.0);

    vec4  sunDirAngle   = _eclipseGetBodyDirAngle(uEclipseSun, position);
    float sunSolidAngle = ECLIPSE_PI * sunDirAngle.w * sunDirAngle.w;

    for (int i = 0; i < uEclipseNumOccluders; ++i) {

      vec4  bodyDirAngle = _eclipseGetBodyDirAngle(uEclipseOccluders[i], position);
      float sunBodyDist  = _eclipseGetAngle(sunDirAngle.xyz, bodyDirAngle.xyz);

      float intersect = 0;

      if (uEclipseMode == 5) {
        intersect = _eclipseGetCircleIntersection(sunDirAngle.w, bodyDirAngle.w, sunBodyDist);
      } else if (uEclipseMode == 6) {
        intersect = _eclipseGetCapIntersectionApprox(sunDirAngle.w, bodyDirAngle.w, sunBodyDist);
      } else {
        intersect = _eclipseGetCapIntersection(sunDirAngle.w, bodyDirAngle.w, sunBodyDist);
      }

      light *= (sunSolidAngle - clamp(intersect, 0.0, sunSolidAngle)) / sunSolidAngle;
    }

    return light;
  }

  // Texture Lookups.
  vec3 light = vec3(1.0);
  for (int i = 0; i < uEclipseNumOccluders; ++i) {
    light *= _eclipseShadowMapLookup(i, position);
  }

  return light;
}

#endif // CS_ECLIPSE_SHADOWS_GLSL