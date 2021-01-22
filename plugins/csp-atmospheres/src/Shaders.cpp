////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Atmosphere.hpp"

namespace csp::atmospheres {

const char* AtmosphereRenderer::cAtmosphereVert = R"(
  #version 330

  // inputs
  layout(location = 0) in vec2 vPosition;

  // uniforms
  uniform mat4 uMatInvMV;
  uniform mat4 uMatInvP;

  // outputs
  out VaryingStruct {
    vec3 vRayDir;
    vec3 vRayOrigin;
    vec2 vTexcoords;
  } vsOut;

  void main() {
    mat4 testInvMV = uMatInvMV;
    testInvMV[3] = vec4(0, 0, 0, 1);

    mat4 testInvMVP = testInvMV * uMatInvP;

    // get camera position in model space
    vsOut.vRayOrigin = uMatInvMV[3].xyz;

    // get ray direction model space
    vsOut.vRayDir = (testInvMVP * vec4(vPosition, 0, 1)).xyz;

    // for lookups in the depth and color buffers
    vsOut.vTexcoords = vPosition * 0.5 + 0.5;

    // no tranformation here since we draw a full screen quad
    gl_Position = vec4(vPosition, 0, 1);
  }
)";

// needs to be splitted because MSVC doesn't like long strings
const char* AtmosphereRenderer::cAtmosphereFrag0 = R"(
  #version 330

  // inputs
  in VaryingStruct {
    vec3 vRayDir;
    vec3 vRayOrigin;
    vec2 vTexcoords;
  } vsIn;

  // uniforms
  #if HDR_SAMPLES > 0
    uniform sampler2DMS uColorBuffer;
    uniform sampler2DMS uDepthBuffer;
  #else
    uniform sampler2D uColorBuffer;
    uniform sampler2D uDepthBuffer;
  #endif

  uniform sampler2D uCloudTexture;
  uniform mat4      uMatInvMVP;
  uniform mat4      uMatInvMV;
  uniform mat4      uMatInvP;
  uniform mat4      uMatMV;
  uniform vec3      uSunDir;
  uniform float     uSunIntensity;
  uniform float     uWaterLevel;
  uniform float     uCloudAltitude;
  uniform float     uAmbientBrightness;
  uniform float     uFarClip;

  // shadow stuff
  uniform sampler2DShadow uShadowMaps[5];
  uniform mat4            uShadowProjectionViewMatrices[5];
  uniform int             uShadowCascades;

  // outputs
  layout(location = 0) out vec3 oColor;

  // constants
  const float PI = 3.14159265359;
  const vec3  BR = vec3(BETA_R_0,BETA_R_1,BETA_R_2);
  const vec3  BM = vec3(BETA_M_0,BETA_M_1,BETA_M_2);

  // for a given cascade and view space position, returns the lookup coordinates
  // for the corresponding shadow map
  vec3 GetShadowMapCoords(int cascade, vec3 position) {
    vec4 smap_coords = uShadowProjectionViewMatrices[cascade] * vec4(position, 1.0);
    return (smap_coords.xyz / smap_coords.w) * 0.5 + 0.5;
  }

  // returns the best cascade containing the given view space position
  int GetCascade(vec3 position) {
    for (int i=0; i<uShadowCascades; ++i) {
      vec3 coords = GetShadowMapCoords(i, position);

      if (coords.x > 0 && coords.x < 1 && 
          coords.y > 0 && coords.y < 1 &&
          coords.z > 0 && coords.z < 1)
      {
        return i;
      }
    }

    return -1;
  } 

  // returns the amount of shadowing going on at the given view space position
  float GetShadow(vec3 position) {
    int cascade = GetCascade(position);

    if (cascade < 0) {
      return 1.0;
    }

    vec3 coords = GetShadowMapCoords(cascade, position);

    float shadow = 0;
    float size = 0.005;

    for(int x=-1; x<=1; x++) {
      for(int y=-1; y<=1; y++) {
        vec2 off = vec2(x,y)*size;

        // Dynamic array lookups are not supported in OpenGL 3.3
        if      (cascade == 0) shadow += texture(uShadowMaps[0], coords - vec3(off, 0.00002));
        else if (cascade == 1) shadow += texture(uShadowMaps[1], coords - vec3(off, 0.00002));
        else if (cascade == 2) shadow += texture(uShadowMaps[2], coords - vec3(off, 0.00002));
        else if (cascade == 3) shadow += texture(uShadowMaps[3], coords - vec3(off, 0.00002));
        else                   shadow += texture(uShadowMaps[4], coords - vec3(off, 0.00002));
      }
    }

    return shadow / 9.0;
  }

  // returns the probability of scattering
  // based on the cosine (c) between in and out direction and the anisotropy (g)
  //
  //            3 * (1 - g*g)               1 + c*c
  // phase = -------------------- * -----------------------
  //          8 * PI * (2 + g*g)     (1 + g*g - 2*g*c)^1.5
  //
  float GetPhase(float fCosine, float fAnisotropy) {
    float fAnisotropy2 = fAnisotropy * fAnisotropy;
    float fCosine2     = fCosine * fCosine;

    float a = (1.0 - fAnisotropy2) * (1.0 + fCosine2);
    float b =  1.0 + fAnisotropy2 - 2.0 * fAnisotropy * fCosine;

    b *= sqrt(b);
    b *= 2.0 + fAnisotropy2;

    return 3.0/(8.0*PI) * a/b;
  }

  // compute the density of the atmosphere for a given model space position
  // returns the rayleigh density as x component and the mie density as Y
  vec2 GetDensity(vec3 vPos) {
    float fHeight = max(0.0, length(vPos) - 1.0 + HEIGHT_ATMO);
    return exp(vec2(-fHeight)/vec2(HEIGHT_R, HEIGHT_M));
  }

  // returns the optical depth between two points in model space
  // The ray is defined by its origin and direction. The two points are defined
  // by two T parameters along the ray. Two values are returned, the rayleigh
  // depth and the mie depth.
  vec2 GetOpticalDepth(vec3 vRayOrigin, vec3 vRayDir, float fTStart, float fTEnd) {
    float fStep = (fTEnd - fTStart) / SECONDARY_RAY_STEPS;
    vec2 vSum = vec2(0.0);

    for (int i=0; i<SECONDARY_RAY_STEPS; i++) {
      float fTCurr = fTStart + (i+0.5)*fStep;
      vec3  vPos = vRayOrigin + vRayDir * fTCurr;
      vSum += GetDensity(vPos);
    }

    return vSum * fStep;
  }

  // calculates the extinction based on an optical depth
  vec3 GetExtinction(vec2 vOpticalDepth) {
    return exp(-BR*vOpticalDepth.x-BM*vOpticalDepth.y);
  }

  // vec3 GetExtinction(vec3 vRayOrigin, vec3 vRayDir)
  // {
  //     float angle = dot(normalize(vRayOrigin), vRayDir);
  //     float altitude = max(0.0, length(vRayOrigin) - 1.0 + HEIGHT_ATMO);
  //     float u = acos(angle) / (98.0 / 180.0 * PI);
  //     float v = altitude / HEIGHT_ATMO;
  //     return texture(uTransmittanceTexture, vec2(u, v)).rgb;
  // }


  // returns the irradiance for the current pixel
  // This is based on the color buffer and the extinction of light.
  vec3 GetExtinction(vec3 vRayOrigin, vec3 vRayDir, float fTStart, float fTEnd) {
    vec2 vOpticalDepth = GetOpticalDepth(vRayOrigin, vRayDir, fTStart, fTEnd);
    return GetExtinction(vOpticalDepth);

    // With precomputed transmittance, this code should be used. As there is no
    // significant performance gain, the code is kept here only for reference
    // vec3 entry = vRayOrigin + vRayDir*fTStart;
    // vec3 exit = vRayOrigin + vRayDir*fTEnd;
    // vec3 extinctionZenith = clamp(GetExtinction(entry, vRayDir) / GetExtinction(exit, vRayDir), vec3(0), vec3(1));
    // vec3 extinctionNadir = clamp(GetExtinction(exit, -vRayDir) / GetExtinction(entry, -vRayDir), vec3(0), vec3(1));
    // float angle = dot(normalize(vRayOrigin), vRayDir);
    // return mix(extinctionNadir, extinctionZenith, clamp((angle+0.01)/0.02, 0.0, 1.0));
  }

  // compute intersections with the atmosphere
  // two T parameters are returned -- if no intersection is found, the first will
  // larger than the second
  vec2 IntersectSphere(vec3 vRayOrigin, vec3 vRayDir, float fRadius) {
    float b = dot(vRayOrigin, vRayDir);
    float c = dot(vRayOrigin, vRayOrigin) - fRadius*fRadius;
    float fDet = b * b - c;

    if (fDet < 0.0) {
      return vec2(10000, -10000);
    }

    fDet = sqrt(fDet);
    return vec2(-b-fDet, -b+fDet);
  }

  vec2 IntersectAtmosphere(vec3 vRayOrigin, vec3 vRayDir) {
    return IntersectSphere(vRayOrigin, vRayDir, 1.0);
  }

  vec2 IntersectPlanetsphere(vec3 vRayOrigin, vec3 vRayDir) {
    return IntersectSphere(vRayOrigin, vRayDir, 1.0-HEIGHT_ATMO);
  }

  vec2 GetLngLat(vec3 vPosition) {
    vec2 result = vec2(-2);

    if (vPosition.z != 0.0) {
      result.x = atan(vPosition.x / vPosition.z);

      if (vPosition.z < 0 && vPosition.x < 0) {
        result.x -= PI;
      }

      if (vPosition.z < 0 && vPosition.x >= 0) {
        result.x += PI;
      }
    } else if (vPosition.x == 0) {
      result.x = 0.0;
    } else if (vPosition.x < 0) {
      result.x = -PI * 0.5;
    } else {
      result.x = PI * 0.5;
    }

    // geocentric latitude of the input point
    result.y = asin(vPosition.y / length(vPosition));

    return result;
  }

  float SRGBtoLINEAR(float srgbIn) {
    float bLess = step(0.04045, srgbIn);
    return mix( srgbIn/12.92, pow((srgbIn+0.055)/1.055, 2.4), bLess);
  }

  float SampleCloudDensity(vec3 vPosition) {
    vec2 lngLat = GetLngLat(vPosition);
    vec2 texCoords = vec2(lngLat.x / (2*PI) + 0.5, 1.0 - lngLat.y / PI + 0.5);

    #if ENABLE_HDR
      return SRGBtoLINEAR(texture(uCloudTexture, texCoords).r);
    #else
      return texture(uCloudTexture, texCoords).r;
    #endif
  }

  vec4 SampleCloudColor(vec3 vRayOrigin, vec3 vRayDir, vec3 vSunDir, float fTIntersection) {
    vec3 point = vRayOrigin + vRayDir * fTIntersection;
    vec2 sunStartEnd = IntersectAtmosphere(point, vSunDir);
    vec3 extinction = GetExtinction(GetOpticalDepth(point, vSunDir, 0, sunStartEnd.y)
                                  + GetOpticalDepth(vRayOrigin, vRayDir, 0, fTIntersection));
    float density = SampleCloudDensity(point);

    return vec4(extinction * density * uSunIntensity, density);
  }

  vec4 GetCloudColor(vec3 vRayOrigin, vec3 vRayDir, vec3 vSunDir, float fOpaqueDepth) {
    vec4 result = vec4(0.0);
    float thickness = uCloudAltitude * 0.2;
    int samples = 10;

    for (int i=0; i<samples; ++i) {
      float altitude = 1.0-HEIGHT_ATMO + uCloudAltitude + i * thickness / samples;

      vec2 vIntersections = IntersectSphere(vRayOrigin, vRayDir, altitude);

      float fac = 1.0;

      // reduce cloud opacity when end point is very close to planet surface
      fac *= clamp(abs(fOpaqueDepth-vIntersections.x)*1000, 0, 1);
      fac *= clamp(abs(fOpaqueDepth-vIntersections.y)*1000, 0, 1);

      // reduce cloud opacity when start point is very close to cloud surface
      fac *= clamp(abs(vIntersections.x)*100, 0, 1);
      fac *= clamp(abs(vIntersections.y)*100, 0, 1);

      if (vIntersections.y > 0 && vIntersections.x < vIntersections.y) {
        if (vIntersections.x > 0 && vIntersections.x < fOpaqueDepth) {
          // hits from above
          result += SampleCloudColor(vRayOrigin, vRayDir, vSunDir, vIntersections.x) * fac;

        } else if (vIntersections.y < fOpaqueDepth) {
          // hits from below
          result += SampleCloudColor(vRayOrigin, vRayDir, vSunDir, vIntersections.y) * fac;
        }
      }
    }

    return result / samples;
  }

  float GetCloudShadow(vec3 vRayOrigin, vec3 vRayDir, float fTStart, float fTEnd) {
    float altitude = 1.0-HEIGHT_ATMO + uCloudAltitude;

    vec2 vIntersections = IntersectSphere(vRayOrigin, vRayDir, altitude);

    float fac = 1.0;

    // reduce cloud opacity when end point is very close to planet surface
    fac *= clamp(abs(fTEnd-vIntersections.x)*1000, 0, 1);
    fac *= clamp(abs(fTEnd-vIntersections.y)*1000, 0, 1);

    // reduce cloud opacity when start point is very close to cloud surface
    fac *= clamp(abs(fTStart-vIntersections.x)*1000, 0, 1);
    fac *= clamp(abs(fTStart-vIntersections.y)*1000, 0, 1);

    if (vIntersections.y > 0 && vIntersections.x < vIntersections.y) {
      if (vIntersections.x > fTStart && vIntersections.x < fTEnd) {
        // hits from above
        return fac * SampleCloudDensity(vRayOrigin + vRayDir * vIntersections.x);
      } else if (vIntersections.y < fTEnd) {
        // hits from below
        return fac * SampleCloudDensity(vRayOrigin + vRayDir * vIntersections.y);
      }
    }
  }

  // very basic tone mapping
  vec3 ToneMapping(vec3 color) {
    #if ENABLE_TONEMAPPING
      color = clamp(EXPOSURE * color, 0.0, 1.0);
      color = pow(color, vec3(1.0 / GAMMA));
    #endif

    return color;
  }


  )";

// needs to be splitted because MSVC doesn't like long strings
const char* AtmosphereRenderer::cAtmosphereFrag1 = R"(
  // Returns the depth at the current pixel. If multisampling is used, we take the minimum depth.
  float GetDepth() {
    #if HDR_SAMPLES > 0
      float depth = 1.0;
      for (int i = 0; i < HDR_SAMPLES; ++i) {
        depth = min(depth, texelFetch(uDepthBuffer, ivec2(vsIn.vTexcoords * textureSize(uDepthBuffer)), i).r);
      }
      return depth;
    #else
      return texture(uDepthBuffer, vsIn.vTexcoords).r;
    #endif
  }

  // Returns the background color at the current pixel. If multisampling is used, we take the average color.
  vec3 GetLandColor() {
    #if HDR_SAMPLES > 0
      vec3 color = vec3(0.0);
      for (int i = 0; i < HDR_SAMPLES; ++i) {
        color += texelFetch(uColorBuffer, ivec2(vsIn.vTexcoords * textureSize(uColorBuffer)), i).rgb;
      }
      return color / HDR_SAMPLES;
    #else
      return texture(uColorBuffer, vsIn.vTexcoords).rgb;
    #endif
  }

  // returns the color of the incoming light for any direction and position
  // The ray is defined by its origin and direction. The two points are defined
  // by two T parameters along the ray. Everything is in model space.
  vec3 GetInscatter(vec3 vRayOrigin, vec3 vRayDir, float fTStart,
                    float fTEnd, bool bHitsSurface, vec3 vLightDir) {
    // we do not always distribute samples evenly:
    //  - if we do hit the planet's surface, we sample evenly
    //  - if the planet surface is not hit, the sampling density depends on 
    //    start height, if we are close to the surface, we will  sample more
    //    at the beginning of the ray where there is more dense atmosphere
    float fstartHeight = clamp((length(vRayOrigin + vRayDir * fTStart) - 1.0 + HEIGHT_ATMO)/HEIGHT_ATMO, 0.0, 1.0);
    const float fMaxExponent = 3.0;
    float fExponent = 1.0;

    if (!bHitsSurface) {
      fExponent = (1.0 - fstartHeight) * (fMaxExponent - 1.0) + 1.0;
    }

    float fDist = (fTEnd - fTStart);
    vec3 sumR   = vec3(0.0);
    vec3 sumM   = vec3(0.0);

    for (float i=0; i<PRIMARY_RAY_STEPS; i++) {
      float fTSegmentBegin  = fTStart + pow((i+0.0)/(PRIMARY_RAY_STEPS), fExponent)*fDist;
      float fTMid           = fTStart + pow((i+0.5)/(PRIMARY_RAY_STEPS), fExponent)*fDist;
      float fTSegmentEnd    = fTStart + pow((i+1.0)/(PRIMARY_RAY_STEPS), fExponent)*fDist;

      vec3  vPos      = vRayOrigin + vRayDir * fTMid;
      float fTSunExit = IntersectAtmosphere(vPos, vLightDir).y;

      // check if we are in shadow - in this case we do not need to sample
      // in sun direction
      float shadow = 1.0;

      #if USE_SHADOWMAP
        vec3 vPosVS = (uMatMV * vec4(vPos, 1.0)).xyz;
        shadow = 0.8 * GetShadow(vPosVS) + 0.2;
      #endif

      vec2 vOpticalDepth    = GetOpticalDepth(vRayOrigin, vRayDir, fTStart, fTMid);
      vec2 vOpticalDepthSun = GetOpticalDepth(vPos, vLightDir, 0, fTSunExit);
      vec3 vExtinction      = GetExtinction(vOpticalDepthSun+vOpticalDepth);

      // With precomputed transmittance, this code should be used. As there is no
      // significant performance gain, the code is kept here only for reference
      // vec3 vExtinction      = GetExtinction(vRayOrigin, vRayDir, fTStart, fTMid) * 
      //                         GetExtinction(vPos, vLightDir, 0, fTSunExit);

      vec2 vDensity         = GetDensity(vPos);

      sumR += vExtinction*vDensity.x * (fTSegmentEnd - fTSegmentBegin) * shadow;
      sumM += vExtinction*vDensity.y * (fTSegmentEnd - fTSegmentBegin) * shadow;
    }

    float fCosine   = dot(vRayDir, vLightDir);
    vec3 vInScatter = sumR * BR * GetPhase(fCosine, ANISOTROPY_R) +
                      sumM * BM * GetPhase(fCosine, ANISOTROPY_M);

    return ToneMapping(uSunIntensity * vInScatter);
  }

  // returns the model space distance to the surface of the depth buffer at the
  // current pixel, or 10 if there is nothing in the depth buffer
  float GetOpaqueDepth() {
    float fDepth = GetDepth();

    #if USE_LINEARDEPTHBUFFER

      // We need to return a distance which is guaranteed to be larger
      // than the largest ray length possible. As the atmosphere has a
      // radius of 1.0, 1000000 is more than enough.
      if (fDepth == 1) return 1000000.0;

      float linearDepth = fDepth * uFarClip;
      vec4 posFarPlane = uMatInvP * vec4(2.0*vsIn.vTexcoords-1, 1.0, 1.0);
      vec3 posVS = normalize(posFarPlane.xyz) * linearDepth;

      return length(vsIn.vRayOrigin - (uMatInvMV * vec4(posVS, 1.0)).xyz);

    #else

      vec4  vPos = uMatInvMVP * vec4(2.0*vsIn.vTexcoords-1, 2*fDepth-1, 1);
      return length(vsIn.vRayOrigin - vPos.xyz / vPos.w);

    #endif
  }
  
  // crops the intersections to the view ray
  bool GetViewRay(vec2 vIntersections, float fOpaqueDepth, out vec2 vStartEnd) {
    if (vIntersections.x > vIntersections.y) {
      // ray does not actually hit the atmosphere
      return false;
    }

    if (vIntersections.y < 0) {
      // ray does not actually hit the atmosphere; exit is behind camera
      return false;
    }

    if (vIntersections.x > fOpaqueDepth) {
      // something is in front of the atmosphere
      return false;
    }

    // if camera is inside of atmosphere, advance ray start to camera
    vStartEnd.x = max(0, vIntersections.x);

    // if something blocks the ray's path, move its end to the object
    vStartEnd.y = min(fOpaqueDepth, vIntersections.y);

    return true;
  }

  // returns a hard-coded color scale for a given ocean depth.
  // Could be configurable in future.
  vec4 GetWaterShade(float v) {
    const float steps[5] = float[](0.0, 0.01, 0.02, 0.2, 1.0);
    const vec4 colors[5] = vec4[](
      vec4(1, 1, 1, 0.0),
      vec4(0.2, 0.8, 0.9, 0.0),
      vec4(0.2, 0.3, 0.4, 0.4),
      vec4(0.1, 0.2, 0.3, 0.8),
      vec4(0.03, 0.05, 0.1, 0.9)
    ); 

    for (int i=0; i<4; ++i) {
      if (v <= steps[i+1]) 
      return mix(colors[i], colors[i+1], vec4(v - steps[i])/(steps[i+1]-steps[i]));
    }
  }

  // returns an artifical ocean color based on the water depth
  vec3 GetWaterColor(vec3 vRayOrigin, vec3 vRayDir, vec2 vStartEnd) {
    // sub-water surface
    vec3 color = GetLandColor();

    vec3 surface = vRayOrigin + vRayDir * vStartEnd.x;
    vec3 normal = normalize(surface);
    float specular = pow(max(dot(vRayDir, reflect(uSunDir, normal)), 0.0), 10)*0.2;
    specular += pow(max(dot(vRayDir, reflect(uSunDir, normal)), 0.0), 50)*0.2;
    specular *= uSunIntensity;

    #if !ENABLE_HDR
      // For non-hdr rendering, the specular needs to be darkend a little.
      specular *= 0.3;
    #endif

    float depth = clamp((vStartEnd.y - vStartEnd.x)*1000, 0.0, 1.0);
    vec4 water = GetWaterShade(depth);
    color = mix(color, water.rgb, water.a) + water.a * specular;

    return color;
  }

  // returns either the result of GetWaterColor or GetLandColor, based on an
  // intersection test.
  vec3 GetBaseColor(vec3 vRayOrigin, vec3 vRayDir, inout float fOpaqueDepth) {
    #if DRAW_WATER
      vec2 vIntersections = IntersectSphere(vRayOrigin, vRayDir, 1.0-HEIGHT_ATMO + uWaterLevel);

      vec2 vStartEnd;
      bool bHitsWater = GetViewRay(vIntersections, fOpaqueDepth, vStartEnd);

      if (bHitsWater) {
        fOpaqueDepth = vStartEnd.x;
        return GetWaterColor(vRayOrigin, vRayDir, vStartEnd);
      } 
    #endif
    
    return GetLandColor();
  }

  void main() {
    vec3 vRayDir = normalize(vsIn.vRayDir);

    // sample depth from the depth buffer
    float fOpaqueDepth = GetOpaqueDepth();

    // get the color of the planet, can be land or ocean
    // if it is ocean, fOpaqueDepth will be increased towards the ocean surface
    oColor = GetBaseColor(vsIn.vRayOrigin, vRayDir, fOpaqueDepth);

    // multiply the surface color with the extinction in light direction
    vec3 surfacePoint = vsIn.vRayOrigin + vRayDir * fOpaqueDepth;
    vec2 sunStartEnd = IntersectAtmosphere(surfacePoint, uSunDir);

    if (sunStartEnd.x < sunStartEnd.y && sunStartEnd.y > 0) {
      vec3 sunExtinction = GetExtinction(surfacePoint, uSunDir, max(0, sunStartEnd.x), sunStartEnd.y);
      
      #if USE_CLOUDMAP
        // add cloud shadow to the surface color
        float cloudShadow = 1.0 - GetCloudShadow(surfacePoint, uSunDir, max(0, sunStartEnd.x), sunStartEnd.y);
        sunExtinction *= cloudShadow;
      #endif

      oColor *= mix(sunExtinction, vec3(1), uAmbientBrightness);
    }

    // vIntersections.x and vIntersections.y are the distances from the ray
    // origin to the intersections of the line defined by the ray direction
    // and the ray origin with the atmosphere boundary (vIntersections.x may
    // be negative if it is behind the origin).
    vec2 vIntersections = IntersectAtmosphere(vsIn.vRayOrigin, vRayDir);

    // vT.x and vT.y are the distances to the actual start and end point of the
    // intersection of the ray with the atmosphere. vT.x will be zero if the
    // origin is inside the atmosphere; vT.y will be smaller than
    // vIntersections.y if there is an occluder in th atmosphere. Overall the
    // following unequality will hold:
    // vIntersections.x <= vT.x < vT.y <= vIntersections.y
    // This function may discard this fragment if no valid ray was generated.
    vec2 vStartEnd;
    bool bHitsAtmosphere = GetViewRay(vIntersections, fOpaqueDepth, vStartEnd);
    bool bHitsSurface = (fOpaqueDepth == vStartEnd.y);

    // now get the actual color
    if (bHitsAtmosphere) {
      #if USE_CLOUDMAP
        // add clouds themselves
        vec4 cloudColor = GetCloudColor(vsIn.vRayOrigin, vRayDir, uSunDir, fOpaqueDepth);
        oColor *= (1-cloudColor.a);
        oColor = mix(oColor, vec3(0.8), cloudColor.a * uAmbientBrightness);
      #endif

      oColor *= GetExtinction(vsIn.vRayOrigin, vRayDir, vStartEnd.x, vStartEnd.y);
      oColor += GetInscatter(vsIn.vRayOrigin, vRayDir, vStartEnd.x, vStartEnd.y, bHitsSurface, uSunDir);

      // add clouds themselves
      #if USE_CLOUDMAP
        #if ENABLE_HDR
          oColor += cloudColor.rgb * (1 - uAmbientBrightness) * 0.3;
        #else
          // For non-hdr rendering, the clouds need to be darkend a little more.
          oColor += cloudColor.rgb * (1 - uAmbientBrightness) * 0.1;
        #endif
      #endif
    }

    // sun position ----------------------------------------------------------
    #if DRAW_SUN
      #if HDR_SAMPLES > 0
        vec2 vTexcoords = vsIn.vTexcoords * textureSize(uDepthBuffer);
      #else
        vec2 vTexcoords = vsIn.vTexcoords * textureSize(uDepthBuffer, 0);
      #endif
      float fDepth = GetDepth();

      if (fDepth == 1.0) {
        float fSunAngle = max(0,dot(vRayDir, uSunDir));
        // glow
        oColor += 0.1*vec3(pow(fSunAngle, 100));
        oColor += 0.3*vec3(pow(fSunAngle, 500));
        oColor += 2.0*vec3(pow(fSunAngle, 47000));
      }
    #endif
  }
)";
} // namespace csp::atmospheres
