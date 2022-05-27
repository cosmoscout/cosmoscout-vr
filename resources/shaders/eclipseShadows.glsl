#ifndef CS_ECLIPSE_SHADOWS_GLSL
#define CS_ECLIPSE_SHADOWS_GLSL

const int   ECLIPSE_MAX_BODIES = 8;
const float ECLIPSE_PI         = 3.14159265358979323846;

uniform vec4      uEclipseSun;
uniform int       uEclipseNumOccluders;
uniform vec4      uEclipseOccluders[ECLIPSE_MAX_BODIES];
uniform sampler2D uEclipseShadowMaps[ECLIPSE_MAX_BODIES];

// -------------------------------------------------------------------------------------------------
// ----------------------------------------- Intersection Math -------------------------------------
// -------------------------------------------------------------------------------------------------

// Returns the surface area of a circle.
float _eclipseGetCircleArea(float r) {
  return ECLIPSE_PI * r * r;
}

float _eclipseGetCircleIntersection(float radiusA, float radiusB, float centerDistance) {

  // No intersection
  if (centerDistance >= radiusA + radiusB) {
    return 0.0;
  }

  // One circle fully in the other (total eclipse or annular)
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

// This returns basically acos(dot(v1, v2)), but has less floating point errors.
float _eclipseGetAngle(vec3 v1, vec3 v2) {
  return 2.0 * asin(0.5 * length(v1 - v2));
}

vec4 _eclipseGetBodyDirAngle(vec4 body, vec3 position) {
  vec3  bodyPos   = body.xyz - position;
  float bodyDist  = length(bodyPos);
  vec3  bodyDir   = bodyPos / bodyDist;
  float bodyAngle = asin(body.w / bodyDist);

  return vec4(bodyDir, bodyAngle);
}

vec3 getEclipseShadow(vec3 position) {

  vec3 light = vec3(1.0);

  // -----------------------------------------------------------------------------------------------
  // ------------------------------------- Debug Mode ----------------------------------------------
  // -----------------------------------------------------------------------------------------------

#if ECLIPSE_MODE == 1
  vec4 sunDirAngle = _eclipseGetBodyDirAngle(uEclipseSun, position);

  for (int i = 0; i < uEclipseNumOccluders; ++i) {

    vec4  bodyDirAngle = _eclipseGetBodyDirAngle(uEclipseOccluders[i], position);
    float sunBodyDist  = _eclipseGetAngle(sunDirAngle.xyz, bodyDirAngle.xyz);

    if (sunDirAngle.w < bodyDirAngle.w - sunBodyDist) {
      light *= vec3(1.0, 0.5, 0.5); // Total eclipse.
    } else if (sunBodyDist < sunDirAngle.w - bodyDirAngle.w) {
      light *= vec3(0.5, 1.0, 0.5); // Annular eclipse.
    } else if (sunBodyDist < sunDirAngle.w + bodyDirAngle.w) {
      light *= vec3(0.5, 0.5, 1.0); // Partial eclipse.
    }
  }
#endif

  // -----------------------------------------------------------------------------------------------
  // -------------------- Linear or smoothstep gradient in the Penumbra ----------------------------
  // -----------------------------------------------------------------------------------------------

#if ECLIPSE_MODE == 2 || ECLIPSE_MODE == 3
  for (int i = 0; i < uEclipseNumOccluders; ++i) {
    float rSun = uEclipseSun.w;
    float rOcc = uEclipseOccluders[i].w;

    float d         = length(uEclipseSun.xyz - uEclipseOccluders[i].xyz);
    float dUmbra    = d * rOcc / (rSun - rOcc);
    float dPenumbra = d * rOcc / (rSun + rOcc);

    float mUmbra    = -rOcc / sqrt(dUmbra * dUmbra - rOcc * rOcc);
    float mPenumbra = rOcc / sqrt(dPenumbra * dUmbra - rOcc * rOcc);

    vec3 toOcc        = uEclipseOccluders[i].xyz - position;
    vec3 sunToOccNorm = (uEclipseOccluders[i].xyz - uEclipseSun.xyz) / d;
    vec3 toOccProj    = dot(toOcc, sunToOccNorm) * sunToOccNorm;

    // Get position in shadow space.
    float posX = length(toOccProj);
    float posY = length(toOcc - toOccProj);

    float penumbra = mPenumbra * (posX + dPenumbra);
    float umbra    = abs(mUmbra * (posX - dUmbra));

    float maxDepth = min(1.0, pow(dUmbra / posX, 2.0));
    float fac      = (posY - umbra) / (penumbra - umbra);

#if ECLIPSE_MODE == 3
    fac = smoothstep(0, 1, fac);
#endif

    light *= 1.0 - maxDepth * clamp(1.0 - fac, 0.0, 1.0);
  }
#endif

  // -----------------------------------------------------------------------------------------------
  // --------------------------------- Circle Intersection -----------------------------------------
  // -----------------------------------------------------------------------------------------------

#if ECLIPSE_MODE == 4
  vec4  sunDirAngle = _eclipseGetBodyDirAngle(uEclipseSun, position);
  float sunArea     = _eclipseGetCircleArea(sunDirAngle.w);

  for (int i = 0; i < uEclipseNumOccluders; ++i) {

    vec4  bodyDirAngle = _eclipseGetBodyDirAngle(uEclipseOccluders[i], position);
    float sunBodyDist  = _eclipseGetAngle(sunDirAngle.xyz, bodyDirAngle.xyz);

    float intersect = _eclipseGetCircleIntersection(sunDirAngle.w, bodyDirAngle.w, sunBodyDist);

    light *= (sunArea - clamp(intersect, 0.0, sunArea)) / sunArea;
  }
#endif

  // -----------------------------------------------------------------------------------------------
  // --------------------- Get Eclipse Shadow by Texture Lookups -----------------------------------
  // -----------------------------------------------------------------------------------------------

#if ECLIPSE_MODE == 5 || ECLIPSE_MODE == 6
  const float textureMappingExponent = 1.0;
  const bool  textureIncludesUmbra   = true;

  vec3  toSun        = uEclipseSun.xyz - position;
  float distToSun    = length(toSun);
  float appSunRadius = uEclipseSun.w / distToSun;

  for (int i = 0; i < uEclipseNumOccluders; ++i) {

    vec3  toCaster     = uEclipseOccluders[i].xyz - position;
    float distToCaster = length(toCaster);

    float appOccluderRadius = uEclipseOccluders[i].w / distToCaster;
    float sunBodyDist       = length(toCaster / distToCaster - toSun / distToSun);

#if ECLIPSE_MODE == 5
    appSunRadius      = asin(appSunRadius);
    appOccluderRadius = asin(appOccluderRadius);
    sunBodyDist       = 2.0 * asin(0.5 * sunBodyDist);
#else
    if (appSunRadius > 0.01) {
      appSunRadius = asin(appSunRadius);
    }

    if (appOccluderRadius > 0.01) {
      appOccluderRadius = asin(appOccluderRadius);
    }

    if (sunBodyDist > 0.01) {
      sunBodyDist = 2.0 * asin(0.5 * sunBodyDist);
    }
#endif

    float minOccDist = textureIncludesUmbra ? 0.0 : max(appOccluderRadius - appSunRadius, 0.0);
    float maxOccDist = appSunRadius + appOccluderRadius;

    float x = 1.0 / (appOccluderRadius / appSunRadius + 1.0);
    float y = (sunBodyDist - minOccDist) / (maxOccDist - minOccDist);

    x = pow(x, textureMappingExponent);
    y = 1.0 - pow(1.0 - y, textureMappingExponent);

    if (!textureIncludesUmbra && y < 0) {
      light = vec3(0.0);
    } else if (x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0) {
      light *= texture(uEclipseShadowMaps[i], vec2(x, 1 - y)).rgb;
    }
  }
#endif

  return light;
}

#endif // CS_ECLIPSE_SHADOWS_GLSL