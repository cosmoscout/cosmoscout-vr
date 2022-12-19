#version 330

// inputs
in VaryingStruct {
  vec3 vRayDir;
  vec3 vRayOrigin;
  vec2 vTexcoords;
}
vsIn;

// Returns the sky luminance along the segment from 'camera' to the nearest
// atmosphere boundary in direction 'view_ray', as well as the transmittance
// along this segment.
vec3 GetSkyLuminance(
    vec3 camera, vec3 view_ray, float shadow_length, vec3 sun_direction, out vec3 transmittance);

// Returns the sky luminance along the segment from 'camera' to 'p', as well as
// the transmittance along this segment.
vec3 GetSkyLuminanceToPoint(
    vec3 camera, vec3 p, float shadow_length, vec3 sun_direction, out vec3 transmittance);

// Returns the sun and sky illuminance received on a surface patch located at
// 'p' and whose normal vector is 'normal'.
vec3 GetSunAndSkyIlluminance(vec3 p, vec3 normal, vec3 sun_direction, out vec3 sky_illuminance);

// uniforms
#if HDR_SAMPLES > 0
uniform sampler2DMS uColorBuffer;
uniform sampler2DMS uDepthBuffer;
#else
uniform sampler2D uColorBuffer;
uniform sampler2D uDepthBuffer;
#endif

uniform sampler2D uCloudTexture;
uniform mat4 uMatInvMVP;
uniform mat4 uMatInvMV;
uniform mat4 uMatInvP;
uniform mat4 uMatMV;
uniform mat4 uMatM;
uniform vec3  uSunDir;
uniform float uCloudAltitude;
uniform float uSunIlluminance;

// shadow stuff
uniform sampler2DShadow uShadowMaps[5];
uniform mat4 uShadowProjectionViewMatrices[5];
uniform int  uShadowCascades;

// outputs
layout(location = 0) out vec3 oColor;

ECLIPSE_SHADER_SNIPPET

// constants
const float PI                = 3.14159265359;
const float PLANET_RADIUS     = 6360000.0;
const float ATMOSPHERE_RADIUS = 6420000.0;

// compute intersections with the atmosphere
// two T parameters are returned -- if no intersection is found, the first will
// larger than the second
vec2 IntersectSphere(vec3 vRayOrigin, vec3 vRayDir, float fRadius) {
  float b    = dot(vRayOrigin, vRayDir);
  float c    = dot(vRayOrigin, vRayOrigin) - fRadius * fRadius;
  float fDet = b * b - c;

  if (fDet < 0.0) {
    return vec2(1, -1);
  }

  fDet = sqrt(fDet);
  return vec2(-b - fDet, -b + fDet);
}

vec2 IntersectAtmosphere(vec3 vRayOrigin, vec3 vRayDir) {
  return IntersectSphere(vRayOrigin, vRayDir, ATMOSPHERE_RADIUS);
}

vec2 IntersectPlanetsphere(vec3 vRayOrigin, vec3 vRayDir) {
  return IntersectSphere(vRayOrigin, vRayDir, PLANET_RADIUS);
}

// for a given cascade and view space position, returns the lookup coordinates
// for the corresponding shadow map
vec3 GetShadowMapCoords(int cascade, vec3 position) {
  vec4 smap_coords = uShadowProjectionViewMatrices[cascade] * vec4(position, 1.0);
  return (smap_coords.xyz / smap_coords.w) * 0.5 + 0.5;
}

// returns the best cascade containing the given view space position
int GetCascade(vec3 position) {
  for (int i = 0; i < uShadowCascades; ++i) {
    vec3 coords = GetShadowMapCoords(i, position);

    if (coords.x > 0 && coords.x < 1 && coords.y > 0 && coords.y < 1 && coords.z > 0 &&
        coords.z < 1) {
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
  float size   = 0.005;

  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      vec2 off = vec2(x, y) * size;

      // Dynamic array lookups are not supported in OpenGL 3.3
      if (cascade == 0)
        shadow += texture(uShadowMaps[0], coords - vec3(off, 0.00002));
      else if (cascade == 1)
        shadow += texture(uShadowMaps[1], coords - vec3(off, 0.00002));
      else if (cascade == 2)
        shadow += texture(uShadowMaps[2], coords - vec3(off, 0.00002));
      else if (cascade == 3)
        shadow += texture(uShadowMaps[3], coords - vec3(off, 0.00002));
      else
        shadow += texture(uShadowMaps[4], coords - vec3(off, 0.00002));
    }
  }

  return shadow / 9.0;
}

// Returns the depth at the current pixel. If multisampling is used, we take the minimum depth.
float GetDepth() {
#if HDR_SAMPLES > 0
  float depth = 1.0;
  for (int i = 0; i < HDR_SAMPLES; ++i) {
    depth = min(
        depth, texelFetch(uDepthBuffer, ivec2(vsIn.vTexcoords * textureSize(uDepthBuffer)), i).r);
  }
  return depth;
#else
  return texture(uDepthBuffer, vsIn.vTexcoords).r;
#endif
}

// returns the model space distance to the surface of the depth buffer at the
// current pixel, or 100 if there is nothing in the depth buffer
float GetOpaqueDepth(vec3 vRayOrigin, vec3 vRayDir) {
  float fDepth = GetDepth();

  // If the fragment is really far away, the inverse reverse infinite projection divides by zero.
  // So we add a minimum threshold here.
  fDepth = max(fDepth, 0.0000001);

  vec4  vPos    = uMatInvMVP * vec4(2.0 * vsIn.vTexcoords - 1, 2 * fDepth - 1, 1);
  float msDepth = length(vRayOrigin - vPos.xyz / vPos.w);

  // // If the depth of the next opaque object is verz close to the far end of our depth buffer, we
  // // will get jittering artifacts. That's the case if we are next to a satellite or on a moon and
  // // look towards a planet with an atmosphere. In this case, start and end of the ray through the
  // // atmosphere basically map to the same depth. Therefore, if the depth is really far away
  // (close
  // // to zero) we compute the intersection with the planet analytically and blend to this value
  // // instead. This means, if you are close to a satellite, mountains of the planet below cannot
  // // poke through the atmosphere anymore.
  // const float START_DEPTH_FADE = 0.001;
  // const float END_DEPTH_FADE   = 0.00001;

  // // We are only using the depth approximation if fDepth is smaller than START_DEPTH_FADE and if
  // // the observer is outside of the atmosphere.
  // if (fDepth < START_DEPTH_FADE && length(vRayOrigin) > 1.0) {
  //   vec2  planetIntersections = IntersectPlanetsphere(vRayOrigin, vRayDir);
  //   float simpleDepth         = planetIntersections.y > 0.0 ? planetIntersections.x : 100.0;
  //   return mix(simpleDepth, msDepth,
  //       clamp((fDepth - END_DEPTH_FADE) / (START_DEPTH_FADE - END_DEPTH_FADE), 0.0, 1.0));
  // }

  return msDepth;
}

// Returns the background color at the current pixel. If multisampling is used, we take the average
// color.
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

uniform sampler2D transmittance_texture;
uniform sampler3D scattering_texture;
uniform sampler3D single_mie_scattering_texture;
uniform sampler2D irradiance_texture;

void main() {
  vec3 vRayDir = normalize(vsIn.vRayDir);

  // sample depth from the depth buffer
  float fOpaqueDepth = GetOpaqueDepth(vsIn.vRayOrigin, vRayDir);

  // get the color of the planet, can be land or ocean
  // if it is ocean, fOpaqueDepth will be increased towards the ocean surface
  oColor = GetLandColor();

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
  bool bHitsSurface    = (fOpaqueDepth == vStartEnd.y);

  float shadowLength = 0.0;

  if (bHitsSurface) {
    vec3 skyIlluminance, transmittance;
    vec3 p = vsIn.vRayOrigin + vRayDir * fOpaqueDepth;
    vec3 inScatter =
        GetSkyLuminanceToPoint(vsIn.vRayOrigin, p, shadowLength, uSunDir, transmittance);
    vec3 sunIlluminance = GetSunAndSkyIlluminance(p, uSunDir, uSunDir, skyIlluminance);

    oColor =
        transmittance * oColor * (skyIlluminance + sunIlluminance) / uSunIlluminance + inScatter;
  } else {
    vec3 transmittance;
    vec3 inScatter =
        GetSkyLuminance(vsIn.vRayOrigin, vRayDir, shadowLength, uSunDir, transmittance);

    oColor = transmittance * oColor + inScatter;
  }

  // oColor = texture(irradiance_texture, vsIn.vTexcoords).rgb;
}