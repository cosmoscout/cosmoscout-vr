////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_ECLIPSE_SHADOWS_GLSL
#define CS_ECLIPSE_SHADOWS_GLSL

// This number should match the constant defined in EclipseShadowReceiver.hpp
const int   ECLIPSE_MAX_BODIES = 4;
const float ECLIPSE_PI         = 3.14159265358979323846;

// The first three components of uEclipseSun and the individual uEclipseOccluders are its position,
// the last component contains its radius.
uniform vec4      uEclipseSun;
uniform int       uEclipseNumOccluders;
uniform vec4      uEclipseOccluders[ECLIPSE_MAX_BODIES];
uniform sampler2D uEclipseShadowMaps[ECLIPSE_MAX_BODIES];

// ------------------------------------------------------------------------------- intersection math

// Returns the surface area of a circle.
float _eclipseGetCircleArea(float r) {
  return ECLIPSE_PI * r * r;
}

// Returns the intersetion area of two circles.
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

  float d1 = (radiusA * radiusA + (-radiusB * radiusB + dd)) / (2 * d);
  float d2 = d - d1;

  float fourth = -d2 * sqrt(-d2 * d2 + rrB);
  float third  = rrB * acos(d2 / radiusB) + fourth;
  float second = -d1 * sqrt(-d1 * d1 + rrA) + third;

  return rrA * acos(d1 / radiusA) + second;
}

// ----------------------------------------------------------------------------------------- helpers

// Returns the angle between two normalized vectors. Usually, this would be implemented with
// acos(dot(v1, v2)), but this variant has less floating point issues for small angles.
float _eclipseGetAngle(vec3 v1, vec3 v2) {
  return 2.0 * asin(0.5 * length(v1 - v2));
}

// This helper returns the normalized direction from position to body.xyz as well as the angular
// radius of the body (its cartesian radius is given by body.w) when watched from position.
vec4 _eclipseGetBodyDirAngle(vec4 body, vec3 position) {
  vec3  bodyPos   = body.xyz - position;
  float bodyDist  = length(bodyPos);
  vec3  bodyDir   = bodyPos / bodyDist;
  float bodyAngle = asin(body.w / bodyDist);

  return vec4(bodyDir, bodyAngle);
}

// -------------------------------------------------------------------------------- public interface

// This method returns the relative Sun brightness in the range [0..1] for the given world space
// coordinate. There are multiple ways to compute the eclipse shadow. Which one is chosen, depends
// on the value of ECLIPSE_MODE. These modes are currently supported:
// 0: None                No eclipse shadows at all.
// 1: Debug               Draws the umbra, antumbra and penumbra in different colors.
// 2: Linear              Use a linear falloff in the penumbra and a quatratic in the antumbra.
// 3: Smoothstep          Use a smoothstep falloff in the penumbra and a quatratic in the antumbra.
// 4: CircleIntersection  Use cirlce intersection math to compute the occluded fraction of the Sun.
// 5: Texture             Retrieve the amount of shadowing from a shadow-lookup texture.
// 6: FastTexture         Like above, but with approaximations in the lookup-coordiante computation.
vec3 getEclipseShadow(vec3 position) {

  vec3 light = vec3(1.0);

  // Debug Mode
#if ECLIPSE_MODE == 1

  // Compute direction to and apparant angle of the Sun.
  vec4 sunDirAngle = _eclipseGetBodyDirAngle(uEclipseSun, position);

  for (int i = 0; i < uEclipseNumOccluders; ++i) {

    // Compute direction to and apparant angle of the occluding body as well as the angular
    // separation to the Sun.
    vec4  bodyDirAngle = _eclipseGetBodyDirAngle(uEclipseOccluders[i], position);
    float delta        = _eclipseGetAngle(sunDirAngle.xyz, bodyDirAngle.xyz);

    if (sunDirAngle.w < bodyDirAngle.w - delta) {
      light *= vec3(1.0, 0.5, 0.5); // Total eclipse.
    } else if (delta < sunDirAngle.w - bodyDirAngle.w) {
      light *= vec3(0.5, 1.0, 0.5); // Annular eclipse.
    } else if (delta < sunDirAngle.w + bodyDirAngle.w) {
      light *= vec3(0.5, 0.5, 1.0); // Partial eclipse.
    }
  }
#endif

  // Linear or smoothstep gradient in the Penumbra
#if ECLIPSE_MODE == 2 || ECLIPSE_MODE == 3
  for (int i = 0; i < uEclipseNumOccluders; ++i) {
    float rSun = uEclipseSun.w;
    float rOcc = uEclipseOccluders[i].w;

    // Compute distances to the tips of the umbra and penumbra cones.
    float d         = length(uEclipseSun.xyz - uEclipseOccluders[i].xyz);
    float dUmbra    = d * rOcc / (rSun - rOcc);
    float dPenumbra = d * rOcc / (rSun + rOcc);

    // Compute slopes of the umbra and penumbra cones.
    float mUmbra    = -rOcc / sqrt(dUmbra * dUmbra - rOcc * rOcc);
    float mPenumbra = rOcc / sqrt(dPenumbra * dUmbra - rOcc * rOcc);

    // Project the vector from the occluder to the reciever onto the sun-occluder axis.
    vec3 toOcc        = uEclipseOccluders[i].xyz - position;
    vec3 sunToOccNorm = (uEclipseOccluders[i].xyz - uEclipseSun.xyz) / d;
    vec3 toOccProj    = dot(toOcc, sunToOccNorm) * sunToOccNorm;

    // Get position in shadow space.
    float posX = length(toOccProj);
    float posY = length(toOcc - toOccProj);

    // Distances of the penumbra and umbra cones from the sun-occluder axis at posX.
    float penumbra = mPenumbra * (posX + dPenumbra);
    float umbra    = abs(mUmbra * (posX - dUmbra));

    // Quadratic falloff beyond the end of the umbra.
    float maxDepth = min(1.0, pow(dUmbra / posX, 2.0));

    // Linear falloff in the penumbra.
    float fac = (posY - umbra) / (penumbra - umbra);

#if ECLIPSE_MODE == 3
    fac = smoothstep(0, 1, fac);
#endif

    light *= 1.0 - maxDepth * clamp(1.0 - fac, 0.0, 1.0);
  }
#endif

  // Circle Intersection
#if ECLIPSE_MODE == 4

  // Compute direction to and apparant angle of the Sun as well as the surface area of the Sun's
  // disc.
  vec4  sunDirAngle = _eclipseGetBodyDirAngle(uEclipseSun, position);
  float sunArea     = _eclipseGetCircleArea(sunDirAngle.w);

  for (int i = 0; i < uEclipseNumOccluders; ++i) {

    // Compute direction to and apparant angle of the occluding body as well as the angular
    // separation to the Sun.
    vec4  bodyDirAngle = _eclipseGetBodyDirAngle(uEclipseOccluders[i], position);
    float delta        = _eclipseGetAngle(sunDirAngle.xyz, bodyDirAngle.xyz);

    // Compute the circle intersection.
    float intersect = _eclipseGetCircleIntersection(sunDirAngle.w, bodyDirAngle.w, delta);

    // The light is reduced according to the visible fraction.
    light *= (sunArea - clamp(intersect, 0.0, sunArea)) / sunArea;
  }
#endif

  // Get Eclipse Shadow by Texture Lookups
#if ECLIPSE_MODE == 5 || ECLIPSE_MODE == 6
  const float textureMappingExponent = 1.0;
  const bool  textureIncludesUmbra   = true;

  // Compute direction to and approximate apparant angle of the Sun.
  vec3  toSun        = uEclipseSun.xyz - position;
  float distToSun    = length(toSun);
  float appSunRadius = uEclipseSun.w / distToSun;

  for (int i = 0; i < uEclipseNumOccluders; ++i) {

    // Compute direction to and approximate apparant angle of the occluder.
    vec3  toCaster          = uEclipseOccluders[i].xyz - position;
    float distToCaster      = length(toCaster);
    float appOccluderRadius = uEclipseOccluders[i].w / distToCaster;

    // Compute approximate angular separation between Sun and occluder.
    float delta = length(toCaster / distToCaster - toSun / distToSun);

// In mode 5, we always compute the exact radii and angular separation using the very expensive
// inverse trigonometric function asin. In mode 6 (the "fast" texture mode), we make use of the
// small angle approximation and only evaluate asin if the argument is larger than a predefined
// threshold.
#if ECLIPSE_MODE == 5
    appSunRadius      = asin(appSunRadius);
    appOccluderRadius = asin(appOccluderRadius);
    delta             = 2.0 * asin(0.5 * delta);
#else
    if (appSunRadius > 0.01) {
      appSunRadius = asin(appSunRadius);
    }

    if (appOccluderRadius > 0.01) {
      appOccluderRadius = asin(appOccluderRadius);
    }

    if (delta > 0.01) {
      delta = 2.0 * asin(0.5 * delta);
    }
#endif

    // Compute texture lookup coordinates.
    float minOccDist = textureIncludesUmbra ? 0.0 : max(appOccluderRadius - appSunRadius, 0.0);
    float maxOccDist = appSunRadius + appOccluderRadius;

    float x = 1.0 / (appOccluderRadius / appSunRadius + 1.0);
    float y = (delta - minOccDist) / (maxOccDist - minOccDist);

    x = pow(x, 1.0 / textureMappingExponent);
    y = 1.0 - pow(1.0 - y, 1.0 / textureMappingExponent);

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