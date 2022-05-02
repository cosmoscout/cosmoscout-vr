#ifndef CS_ECLIPSE_SHADOWS_GLSL
#define CS_ECLIPSE_SHADOWS_GLSL

const int    ECLIPSE_MAX_BODIES = 8;
const float  ECLIPSE_PI         = 3.14159265358979323846;
const double ECLIPSE_PI_D       = 3.14159265358979323846;

uniform vec4      uEclipseSun;
uniform int       uEclipseNumOccluders;
uniform vec4      uEclipseOccluders[ECLIPSE_MAX_BODIES];
uniform sampler2D uEclipseShadowMaps[ECLIPSE_MAX_BODIES];

// -------------------------------------------------------------------------------------------------
// ------------------------------- Double Precision Trigonometric Functions ------------------------
// -------------------------------------------------------------------------------------------------

double atand(double y, double x) {
    const double atan_tbl[] = double[](
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
    );

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

    p = fma(fma(fma(fma(fma(fma(fma(fma(fma(fma(p, s,
    atan_tbl[8]), s,
    atan_tbl[7]), s,
    atan_tbl[6]), s,
    atan_tbl[5]), s,
    atan_tbl[4]), s,
    atan_tbl[3]), s,
    atan_tbl[2]), s,
    atan_tbl[1]), s,
    atan_tbl[0]), s * a, a);

    double r = ay > ax ? (1.57079632679489661923LF - p) : p;

    r = x < 0 ?  3.14159265358979323846LF - r : r;
    r = y < 0 ? -r : r;

    return r;
}

double sind(double x) {
    //minimax coefs for sin for 0..pi/2 range
    const double a3 = -1.666666660646699151540776973346659104119e-1LF;
    const double a5 =  8.333330495671426021718370503012583606364e-3LF;
    const double a7 = -1.984080403919620610590106573736892971297e-4LF;
    const double a9 =  2.752261885409148183683678902130857814965e-6LF;
    const double ab = -2.384669400943475552559273983214582409441e-8LF;

    const double m_2_pi = 0.636619772367581343076LF;
    const double m_pi_2 = 1.57079632679489661923LF;

    double y = abs(x * m_2_pi);
    double q = floor(y);
    int quadrant = int(q);

    double t = (quadrant & 1) != 0 ? 1 - y + q : y - q;
    t *= m_pi_2;

    double t2 = t * t;
    double r = fma(fma(fma(fma(fma(ab, t2, a9), t2, a7), t2, a5), t2, a3), t2 * t, t);

    r = x < 0 ? -r : r;

    return (quadrant & 2) != 0 ? -r : r;
}

//cos approximation, error < 5e-11
double cosd(double x) {
    //sin(x + PI/2) = cos(x)
    return sind(x + 1.57079632679489661923LF);
}

/* compute arcsin (a) for a in [-9/16, 9/16] */
double asin_core(double a) {
    double s = a * a;
    double q = s * s;
    double r =      5.5579749017470502e-2LF;
    double t =     -6.2027913464120114e-2LF;
    r = fma (r, q, 5.4224464349245036e-2LF);
    t = fma (t, q, -1.1326992890324464e-2LF);
    r = fma (r, q, 1.5268872539397656e-2LF);
    t = fma (t, q, 1.0493798473372081e-2LF);
    r = fma (r, q, 1.4106045900607047e-2LF);
    t = fma (t, q, 1.7339776384962050e-2LF);
    r = fma (r, q, 2.2372961589651054e-2LF);
    t = fma (t, q, 3.0381912707941005e-2LF);
    r = fma (r, q, 4.4642857881094775e-2LF);
    t = fma (t, q, 7.4999999991367292e-2LF);
    r = fma (r, s, t);
    r = fma (r, s, 1.6666666666670193e-1LF);
    t = a * s;
    r = fma (r, t, a);

    return r;
}

/* Compute arccosine (a), maximum error observed: 1.4316 ulp
   Double-precision factorization of π courtesy of Tor Myklebust
*/
double acosd(double a) {
    double r = (a > 0.0LF) ? -a : a;// avoid modifying the "sign" of NaNs
    if (r > -0.5625LF) {
        /* arccos(x) = pi/2 - arcsin(x) */
        r = fma (9.3282184640716537e-1LF, 1.6839188885261840e+0LF, asin_core(r));
    } else {
        /* arccos(x) = 2 * arcsin (sqrt ((1-x) / 2)) */
        r = 2.0LF * asin_core(sqrt(fma(0.5LF, r, 0.5LF)));
    }
    if (!(a > 0.0LF) && (a >= -1.0LF)) { // avoid modifying the "sign" of NaNs
        /* arccos (-x) = pi - arccos(x) */
        r = fma (1.8656436928143307e+0LF, 1.6839188885261840e+0LF, -r);
    }
    return r;
}

double asind(double a) {
    return (ECLIPSE_PI_D / 2.0LF) - acosd(a);
}


// -------------------------------------------------------------------------------------------------
// ----------------------------------------- Intersection Math -------------------------------------
// -------------------------------------------------------------------------------------------------

// Returns the surface area of a circle.
float _eclipseGetCircleArea(float r) {
  return ECLIPSE_PI * r * r;
}
double _eclipseGetCircleAreaD(double r) {
  return ECLIPSE_PI_D * r * r;
}

// Returns the surface area of a spherical cap on a unit sphere.
float _eclipseGetCapArea(float r) {
  return 2.0 * ECLIPSE_PI * (1.0 - cos(r));
}
double _eclipseGetCapAreaD(double r) {
  return 2.0 * ECLIPSE_PI_D * (1.0 - cosd(r));
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
double _eclipseGetCircleIntersectionD(double radiusA, double radiusB, double centerDistance) {

  // No intersection
  if (centerDistance >= radiusA + radiusB) {
    return 0.0;
  }

  // One circle fully in the other (total eclipse or annular)
  if (min(radiusA, radiusB) <= max(radiusA, radiusB) - centerDistance) {
    return _eclipseGetCircleAreaD(min(radiusA, radiusB));
  }

  double d = centerDistance;

  double rrA = radiusA * radiusA;
  double rrB = radiusB * radiusB;
  double dd  = d * d;

  double d1 = fma(radiusA, radiusA, fma(-radiusB, radiusB, dd)) / (2 * d);
  double d2 = d - d1;

  double fourth = -d2 * sqrt(fma(-d2, d2, rrB));
  double third  = fma(rrB, acosd(d2 / radiusB), fourth);
  double second = fma(-d1, sqrt(fma(-d1, d1, rrA)), third);

  return fma(rrA, acosd(d1 / radiusA), second);
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

  float a = acos(fma(-cosRA, cosRB, cosD) / (sinRA * sinRB));
  float b = acos(fma(-cosD, cosRA, cosRB) / (sinD * sinRA));
  float c = acos(fma(-cosD, cosRB, cosRA) / (sinD * sinRB));

  return 2.0 * (ECLIPSE_PI + fma(-b, cosRA, fma(-c, cosRB, -a)));
}
double _eclipseGetCapIntersectionD(double radiusA, double radiusB, double centerDistance) {

  // No intersection
  if (centerDistance >= radiusA + radiusB) {
    return 0.0;
  }

  // One circle fully in the other
  if (min(radiusA, radiusB) <= max(radiusA, radiusB) - centerDistance) {
    return _eclipseGetCapAreaD(min(radiusA, radiusB));
  }

  double sinD  = sind(centerDistance);
  double cosD  = cosd(centerDistance);
  double sinRA = sind(radiusA);
  double sinRB = sind(radiusB);
  double cosRA = cosd(radiusA);
  double cosRB = cosd(radiusB);

  double a = acosd(fma(-cosRA, cosRB, cosD) / (sinRA * sinRB));
  double b = acosd(fma(-cosD, cosRA, cosRB) / (sinD * sinRA));
  double c = acosd(fma(-cosD, cosRB, cosRA) / (sinD * sinRB));

  return 2.0 * (ECLIPSE_PI_D + fma(-b, cosRA, fma(-c, cosRB, -a)));
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

// This returns basically acos(dot(v1, v2)), but has less floating point errors.
// https://api.semanticscholar.org/CorpusID:118459706
float _eclipseGetAngle(vec3 v1, vec3 v2) {
  float c = dot(v1 - v2, v1 - v2);
  return 2.0 * atan(sqrt(c / (4 - c)));
}
double _eclipseGetAngleD(dvec3 v1, dvec3 v2) {
  double c = dot(v1 - v2, v1 - v2);
  return 2.0 * atand(sqrt(c), sqrt(4 - c));
}

vec4 _eclipseGetBodyDirAngle(vec4 body, vec3 position) {
  vec3  bodyPos   = body.xyz - position;
  float bodyDist  = length(bodyPos);
  vec3  bodyDir   = bodyPos / bodyDist;
  float bodyAngle = asin(body.w / bodyDist);

  return vec4(bodyDir, bodyAngle);
}
dvec4 _eclipseGetBodyDirAngleD(dvec4 body, dvec3 position) {
  dvec3  bodyPos   = body.xyz - position;
  double bodyDist  = length(bodyPos);
  dvec3  bodyDir   = bodyPos / bodyDist;
  double bodyAngle = asind(body.w / bodyDist);

  return dvec4(bodyDir, bodyAngle);
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
  // -------------------------------------- Celestia -----------------------------------------------
  // -----------------------------------------------------------------------------------------------

  // Quaoting a source code comment from Celestia: "All of the eclipse related code assumes that
  // both the caster and receiver are spherical. Irregular receivers will work more or less
  // correctly, but casters that are sufficiently non-spherical will produce obviously incorrect
  // shadows. Another assumption we make is that the distance between the caster and receiver is
  // much less than the distance between the sun and the receiver. This approximation works
  // everywhere in the solar system, and is likely valid for any orbitally stable pair of objects
  // orbiting a star."

  // Also from the source code: "The shadow shadow consists of a circular region of constant depth
  // (maxDepth), surrounded by a ring of linear falloff from maxDepth to zero. For a total eclipse,
  // maxDepth is zero. In reality, the falloff function is much more complex: to calculate the exact
  // amount of sunlight blocked, we need to calculate the a circle-circle intersection area."

  // There seem to be some geometric simplifications in this code - the apparent radii of the bodies
  // are computed by dividing their actual radius by the distance. This is actually not valid for
  // spheres but only for circles. However, the introduced error seems to be very small. There's no
  // noticeable difference to the more complete implementation in the Cosmographia version further
  // below.

  // Based on this code:
  // https://github.com/CelestiaProject/Celestia/blob/master/src/celengine/shadermanager.cpp#L1344
  // https://github.com/CelestiaProject/Celestia/blob/master/src/celengine/shadermanager.cpp#L3811
  // https://github.com/CelestiaProject/Celestia/blob/master/src/celengine/render.cpp#L2969
  // https://github.com/CelestiaProject/Celestia/blob/master/src/celengine/render.cpp#L2377

  #if ECLIPSE_MODE == 2
    for (int i = 0; i < uEclipseNumOccluders; ++i) {
      
      float distToSun    = length(uEclipseSun.xyz - position);
      float appSunRadius = uEclipseSun.w / distToSun;

      float distToCaster      = length(uEclipseOccluders[i].xyz - position);
      float appOccluderRadius = uEclipseOccluders[i].w / distToCaster;

    // The code below is basically the original code from celestia. If we substitute some values, we
    // end up with the version below. This is much easier to read and shows that the umbra and
    // penumbra cones use the same apex angle. This is not the case in reality. However, if the Sun
    // is far away, they become very similar indeed.  
    #if 0
      float penumbraRadius = (1 + appSunRadius / appOccluderRadius) * uEclipseOccluders[i].w;
      float umbraRadius = uEclipseOccluders[i].w * (appOccluderRadius - appSunRadius) / appOccluderRadius;
    #else
      float spread = uEclipseSun.w / distToSun * distToCaster;
      float penumbraRadius = uEclipseOccluders[i].w + spread;
      float umbraRadius =    uEclipseOccluders[i].w - spread;
    #endif

      float maxDepth = min(1.0, pow(appOccluderRadius / appSunRadius, 2.0));

      float umbra   = umbraRadius / penumbraRadius;
      float falloff = maxDepth / max(0.001, 1.0 - abs(umbra));

      // Project the vector from fragment to occluder on the Sun-Occluder ray.
      vec3 toOcc        = uEclipseOccluders[i].xyz - position;
      vec3 sunToOccNorm = normalize(uEclipseOccluders[i].xyz - uEclipseSun.xyz);
      vec3 toOccProj    = dot(toOcc, sunToOccNorm) * sunToOccNorm;

      // Get vertical position in shadow space.
      float posY = length(toOcc - toOccProj);

      // This r is computed quite differently in Celestia. This is due to the fact that eclipse
      // shadows are not computed in worldspace in Celestia but rather in a shadow-local coordinate
      // system.
      float r = 1 - posY / penumbraRadius;

      if (r > 0.0) {
        float shadowR = clamp(r * falloff, 0.0, maxDepth);
        light *= 1 - shadowR;
      }
    }
  #endif

  // -----------------------------------------------------------------------------------------------
  // ----------------------------------- Cosmographia ----------------------------------------------
  // -----------------------------------------------------------------------------------------------

  // Cosmographia (or rather the VESTA library which is used by Cosmographia) performs a very
  // involved computation of the umbra and penumbra cones. In fact, it claims to support ellipsoidal
  // shadow caster by asymmetrical scaling of the shadow matrix. For now, this is difficult to
  // replicate here, however, when compared to the other evaluated solutions, it seems to be the
  // only one which computes the correct apex angles of the cones. To replicate the behavior here,
  // we use out own code to compute the penumbra and umbra radius and use the Cosmosgraphia approach
  // to map this to a shadow value.

  // Yet, it seems to use a linear falloff from the umbra to the penumbra and I do not see a proper
  // falloff handling beyond the end of the umbra.

  // Based on this code:
  // https://github.com/claurel/cosmographia/blob/171462736a30c06594dfc45ad2daf85d024b20e2/thirdparty/vesta/internal/EclipseShadowVolumeSet.cpp
  // https://github.com/claurel/cosmographia/blob/171462736a30c06594dfc45ad2daf85d024b20e2/thirdparty/vesta/ShaderBuilder.cpp#L222
  // https://github.com/claurel/cosmographia/blob/171462736a30c06594dfc45ad2daf85d024b20e2/thirdparty/vesta/UniverseRenderer.cpp#L1980

  #if ECLIPSE_MODE == 3
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
    float umbra    = mUmbra * (posX - dUmbra);

      // As umbra becomes negative beyond the end of the umbra, the results of this code are wrong
      // from this point on.
      light *= clamp((posY - umbra) / (penumbra - umbra), 0.0, 1.0);
    }
  #endif

  // -----------------------------------------------------------------------------------------------
  // ------------------------------------- OpenSpace -----------------------------------------------
  // -----------------------------------------------------------------------------------------------

  // At the moment, the eclipse shadows in OpenSpace seem to be quite basic. They assume a spherical
  // light source as well as spherical shadow casters. There are some geometric simplifications in
  // computing the umbra and penumbra cones - effectively the Sun and the shadow caster are modelled
  // as circles oriented perpendicular to the Sun-Occluder axis. Furthermore, there seems to be no
  // shadow falloff beyond the end of the umbra. The penumbra keeps getting wider and wider, but the
  // center of the shadow volume will always stay black.

  // Based on this code:
  // https://github.com/OpenSpace/OpenSpace/blob/d7d279ea168f5eaa6a0109593360774246699c4e/modules/globebrowsing/shaders/renderer_fs.glsl#L93
  // https://github.com/OpenSpace/OpenSpace/blob/d7d279ea168f5eaa6a0109593360774246699c4e/modules/globebrowsing/src/renderableglobe.cpp#L2086

  #if ECLIPSE_MODE == 4
    for (int i = 0; i < uEclipseNumOccluders; ++i) {
      float distToSun = length(uEclipseSun.xyz - uEclipseOccluders[i].xyz);

      // Project the vector from fragment to occluder on the Sun-Occluder ray.
      vec3  pc             = uEclipseOccluders[i].xyz - position;
      vec3  sc_norm        = (uEclipseOccluders[i].xyz - uEclipseSun.xyz) / distToSun;
      vec3  pc_proj        = dot(pc, sc_norm) * sc_norm;
      float length_pc_proj = length(pc_proj);

      // Compute distance from fragment to Sun-Occluder ray.
      vec3  d        = pc - pc_proj;
      float length_d = length(d);

      // Compute focus point of the penumbra cone. Somewhere in front of the occluder.
      float xp = uEclipseOccluders[i].w * distToSun / (uEclipseSun.w + uEclipseOccluders[i].w);

      // Compute focus point of the umbra cone. Somewhere behind occluder.
      float xu = uEclipseOccluders[i].w * distToSun / (uEclipseSun.w - uEclipseOccluders[i].w);

      // The radius of the penumbra cone, computed with the intercept theorem. This is not really
      // correct, as the tangents at the occluder do not really touch the poles.
      float r_p_pi = uEclipseOccluders[i].w * (length_pc_proj + xp) / xp;
      float r_u_pi = uEclipseOccluders[i].w * (xu - length_pc_proj) / xu;

      if (length_d < r_u_pi) { // umbra

        // The original code uses this:
        // light *= sqrt(r_u_pi / (r_u_pi + pow(length_d, 2.0)));

        // In open space, this is close to zero in most cases, however as we are in the umbra, using
        // exaclty zero seems more correct...
        light *= 0.0;

      } else if (length_d < r_p_pi) { // penumbra

        // This returns a linear falloff from the center of the shadow to the penumbra's edge. Using
        // light *= (length_d - max(0, r_u_pi)) / (r_p_pi - max(0, r_u_pi));
        // would have been better as this decays to zero towards the umbra. Nevertheless, this code
        // still returns a completely black shadow center even behind the end of the umbra...?

        light *= length_d / r_p_pi;
      }
    }
  #endif

  // -----------------------------------------------------------------------------------------------
  // ---------------------------- Various Analytical Approaches ------------------------------------
  // -----------------------------------------------------------------------------------------------

  // 5: Circle Intersection
  // 6: Approximated Spherical Cap Intersection
  // 7: Spherical Cap Intersection
  #if ECLIPSE_MODE == 5 || ECLIPSE_MODE == 6 || ECLIPSE_MODE == 7
    vec4  sunDirAngle = _eclipseGetBodyDirAngle(uEclipseSun, position);
    float sunArea     = _eclipseGetCircleArea(sunDirAngle.w);

    for (int i = 0; i < uEclipseNumOccluders; ++i) {

      vec4  bodyDirAngle = _eclipseGetBodyDirAngle(uEclipseOccluders[i], position);
      float sunBodyDist  = _eclipseGetAngle(sunDirAngle.xyz, bodyDirAngle.xyz);

      float intersect = 0;

      #if ECLIPSE_MODE == 5
        intersect = _eclipseGetCircleIntersection(sunDirAngle.w, bodyDirAngle.w, sunBodyDist);
      #elif ECLIPSE_MODE == 6
        intersect = _eclipseGetCapIntersectionApprox(sunDirAngle.w, bodyDirAngle.w, sunBodyDist);
      #else
        intersect = _eclipseGetCapIntersection(sunDirAngle.w, bodyDirAngle.w, sunBodyDist);
      #endif

      light *= (sunArea - clamp(intersect, 0.0, sunArea)) / sunArea;
    }
  #endif

  // -----------------------------------------------------------------------------------------------
  // ------------------- Various Analytical Approaches (Double Precision) --------------------------
  // -----------------------------------------------------------------------------------------------

  // 8: Circle Intersection (Double Precision)
  // 9: Spherical Cap Intersection (Double Precision)
  #if ECLIPSE_MODE == 8 || ECLIPSE_MODE == 9
    dvec4  sunDirAngle = _eclipseGetBodyDirAngleD(uEclipseSun, position);
    double sunArea     = _eclipseGetCircleAreaD(sunDirAngle.w);

    for (int i = 0; i < uEclipseNumOccluders; ++i) {

      dvec4  bodyDirAngle = _eclipseGetBodyDirAngleD(uEclipseOccluders[i], position);
      double sunBodyDist  = _eclipseGetAngleD(sunDirAngle.xyz, bodyDirAngle.xyz);

      double intersect = 0;

      #if ECLIPSE_MODE == 8
        intersect = _eclipseGetCircleIntersectionD(sunDirAngle.w, bodyDirAngle.w, sunBodyDist);
      #else
        intersect = _eclipseGetCapIntersectionD(sunDirAngle.w, bodyDirAngle.w, sunBodyDist);
      #endif

      light *= float((sunArea - clamp(intersect, 0.0LF, sunArea)) / sunArea);
    }
  #endif

  // -----------------------------------------------------------------------------------------------
  // --------------------- Get Eclipse Shadow by Texture Lookups -----------------------------------
  // -----------------------------------------------------------------------------------------------

  #if ECLIPSE_MODE == 10
    const float textureMappingExponent = 1.0;
    const bool  textureIncludesUmbra   = true;

    vec4  sunDirAngle   = _eclipseGetBodyDirAngle(uEclipseSun, position);
    float sunSolidAngle = ECLIPSE_PI * sunDirAngle.w * sunDirAngle.w;

    for (int i = 0; i < uEclipseNumOccluders; ++i) {

      vec4  bodyDirAngle = _eclipseGetBodyDirAngle(uEclipseOccluders[i], position);
      float sunBodyDist  = _eclipseGetAngle(sunDirAngle.xyz, bodyDirAngle.xyz);

      float minOccDist = textureIncludesUmbra ? 0.0 : max(bodyDirAngle.w - sunDirAngle.w, 0.0);
      float maxOccDist = sunDirAngle.w + bodyDirAngle.w;

      float x = 1.0 / (bodyDirAngle.w / sunDirAngle.w + 1.0);
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

  // -----------------------------------------------------------------------------------------------
  // --------------- Get Eclipse Shadow by Approximated Texture Lookups ----------------------------
  // -----------------------------------------------------------------------------------------------

  #if ECLIPSE_MODE == 11
    const float textureMappingExponent = 1.0;
    const bool  textureIncludesUmbra   = true;

    vec3 toSun = normalize(uEclipseSun.xyz - position);

    for (int i = 0; i < uEclipseNumOccluders; ++i) {

      float distToSun  = length(uEclipseOccluders[i].xyz - uEclipseSun.xyz);
      float appSunRadius = uEclipseSun.w / distToSun;

      float distToCaster      = length(uEclipseOccluders[i].xyz - position);
      float appOccluderRadius = uEclipseOccluders[i].w / distToCaster;

      float sunBodyDist = length((uEclipseOccluders[i].xyz - position) / distToCaster - toSun);
      // float sunBodyDist = sqrt(2.0 - 2.0 * dot((uEclipseOccluders[i].xyz - position) / distToCaster, toSun));
      // float sunBodyDist  = _eclipseGetAngle((uEclipseOccluders[i].xyz - position)/distToCaster, toSun);

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