#ifndef CS_ECLIPSE_SHADOWS_GLSL
#define CS_ECLIPSE_SHADOWS_GLSL

const float ECLIPSE_TEX_SHADOW_EXPONENT = 1.0;
const int   ECLIPSE_MAX_BODIES          = 8;

uniform int       uEclipseMode;
uniform vec4      uEclipseSun;
uniform int       uEclipseNumOccluders;
uniform vec4      uEclipseOccluders[ECLIPSE_MAX_BODIES];
uniform sampler2D uEclipseShadowMaps[ECLIPSE_MAX_BODIES];

vec3 eclipseProjectPointOnRay(vec3 origin, vec3 direction, vec3 p) {
  vec3 ap = p - origin;
  vec3 ab = direction - origin;
  return origin + (dot(ap, ab) / dot(ab, ab)) * ab;
}

vec3 eclipseShadowMapLookup(int i, vec3 position) {
  float sunDistance = length(uEclipseSun.xyz - uEclipseOccluders[i].xyz);
  float sunRadius   = uEclipseSun.w;

  float rOcc   = uEclipseOccluders[i].w;
  float dOcc   = sunDistance / (sunRadius / rOcc + 1);
  float y0     = rOcc / dOcc * sqrt(dOcc * dOcc - rOcc * rOcc);
  float xOcc   = sqrt(rOcc * rOcc - y0 * y0);
  float xUmbra = (sunDistance * rOcc) / (sunRadius - rOcc);
  float xF     = xOcc - dOcc;
  float fac    = y0 / -xF;

  vec3 pSunOcc = eclipseProjectPointOnRay(
      uEclipseOccluders[i].xyz, uEclipseOccluders[i].xyz - uEclipseSun.xyz, position);
  vec2 pos = vec2(distance(pSunOcc, uEclipseOccluders[i].xyz), distance(pSunOcc, position));

  float alphaX = pos.x / (pos.x + xUmbra);
  float alphaY = pos.y / (fac * (pos.x - xF));

  float x = pow(alphaX, 1.0 / ECLIPSE_TEX_SHADOW_EXPONENT);
  float y = alphaY;

  if (x <= 0.0 || x >= 1.0 || y <= 0.0 || y >= 1.0) {
    return vec3(1.0);
  }

  return texture(uEclipseShadowMaps[i], vec2(x, 1 - y)).rgb;
}

vec3 getEclipseShadow(vec3 position) {

  // None.
  if (uEclipseMode == 0) {
    return vec3(1.0);
  }

  // Debug.
  if (uEclipseMode == 1) {
    return vec3(1.0, 0, 0);
  }

  vec3 light = vec3(1.0);
  for (int i = 0; i < uEclipseNumOccluders; ++i) {
    light *= eclipseShadowMapLookup(i, position);
  }

  return light;
}

#endif // CS_ECLIPSE_SHADOWS_GLSL