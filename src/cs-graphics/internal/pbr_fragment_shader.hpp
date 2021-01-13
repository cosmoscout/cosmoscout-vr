////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_PBR_FRAGMENT_SHADER_HPP
#define CS_GRAPHICS_PBR_FRAGMENT_SHADER_HPP

namespace cs::graphics::internal {

const char* GLTF_FRAG = R"( 
//
// This fragment shader defines a reference implementation for Physically Based Shading of
// a microfacet surface material defined by a glTF model.
//
// References:
// [1] Real Shading in Unreal Engine 4
//     http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
// [2] Physically Based Shading at Disney
//     http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
// [3] README.md - Environment Maps
//     https://github.com/KhronosGroup/glTF-WebGL-PBR/#environment-maps
// [4] "An Inexpensive BRDF Model for Physically based Rendering" by Christophe Schlick
//     https://www.cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf
#ifdef USE_TEX_LOD
#extension GL_NV_shadow_samplers_cube : enable
#endif

#define saturate(x) clamp(x, 0.0, 1.0)

out vec4 FragColor;

uniform vec3 u_LightDirection;
uniform vec3 u_LightColor;
uniform bool u_EnableHDR;

#ifdef USE_IBL
uniform samplerCube u_DiffuseEnvSampler;
uniform samplerCube u_SpecularEnvSampler;
uniform sampler2D u_brdfLUT;
uniform float u_IBLIntensity;
uniform mat3 u_IBLrotation;
#endif

#ifdef HAS_BASECOLORMAP
uniform sampler2D u_BaseColorSampler;
#endif
#ifdef HAS_NORMALMAP
uniform sampler2D u_NormalSampler;
#endif
#ifdef HAS_EMISSIVEMAP
uniform sampler2D u_EmissiveSampler;
uniform vec3 u_EmissiveFactor;
#endif
#ifdef HAS_METALROUGHNESSMAP
// Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
// This layout intentionally reserves the 'r' channel for (optional) occlusion map data
uniform sampler2D u_MetallicRoughnessSampler;
#endif
#ifdef HAS_OCCLUSIONMAP
uniform sampler2D u_OcclusionSampler;
uniform float u_OcclusionStrength;
#endif

#ifdef USE_LINEARDEPTHBUFFER
uniform float u_FarClip;
#endif

uniform vec2 u_MetallicRoughnessValues;
uniform vec4 u_BaseColorFactor;
uniform vec3 u_Camera;

in vec3 v_Position;
in vec2 v_UV;

#ifdef HAS_NORMALS
#ifdef HAS_TANGENTS
in mat3 v_TBN;
#else
in vec3 v_Normal;
#endif
#endif

int mipCount = 10;

const float M_PI = 3.141592653589793;
const float c_MinRoughness = 0.04;

vec3 sRGB_to_linear(vec3 c)
{
  return mix(vec3(c * (1.0 / 12.92)),
             pow((c + 0.055)/1.055, vec3(2.4)),
             greaterThanEqual(c, vec3(0.04045)));
}

vec4 SRGBtoLINEAR(vec4 srgbIn)
{
#ifdef MANUAL_SRGB
    vec3 bLess = step(vec3(0.04045),srgbIn.xyz);
    vec3 linOut = mix( srgbIn.xyz/vec3(12.92), pow((srgbIn.xyz+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
    return vec4(linOut,srgbIn.w);;
#else //MANUAL_SRGB
    return srgbIn;
#endif //MANUAL_SRGB
}

// Find the normal for this fragment, pulling either from a predefined normal map
// or from the interpolated mesh normal and tangent attributes.
vec3 getNormal()
{
    // Retrieve the tangent space matrix
#ifndef HAS_TANGENTS
    vec3 pos_dx = dFdx(v_Position);
    vec3 pos_dy = dFdy(v_Position);
    vec3 tex_dx = dFdx(vec3(v_UV, 0.0));
    vec3 tex_dy = dFdy(vec3(v_UV, 0.0));
    vec3 t = (tex_dy.t * pos_dx - tex_dx.t * pos_dy) / (tex_dx.s * tex_dy.t - tex_dy.s * tex_dx.t);

#ifdef HAS_NORMALS
    vec3 ng = normalize(v_Normal);
#else
    vec3 ng = cross(pos_dx, pos_dy);
#endif

    t = normalize(t - ng * dot(ng, t));
    vec3 b = normalize(cross(ng, t));
    mat3 tbn = mat3(t, b, ng);
#else // HAS_TANGENTS
    mat3 tbn = v_TBN;
#endif

#ifdef HAS_NORMALMAP
    vec3 N = texture2D(u_NormalSampler, v_UV).rgb;
    N = normalize(tbn * (2.0 * N - 1.0));
#else
    vec3 N = tbn[2].xyz;
#endif

    return N;
}

float getMetallness()
{
    float metallness = u_MetallicRoughnessValues.x;
#ifdef HAS_METALROUGHNESSMAP
    metallness = texture2D(u_MetallicRoughnessSampler, v_UV).b * metallness;
#endif
    return saturate(metallness);
}

float getRoughness()
{
  float perceptualRoughness = u_MetallicRoughnessValues.y;
#ifdef HAS_METALROUGHNESSMAP
  perceptualRoughness = texture2D(u_MetallicRoughnessSampler, v_UV).g * perceptualRoughness;
#endif
  return clamp(perceptualRoughness, c_MinRoughness, 1.0);
}

// Calculation of the lighting contribution from an optional Image Based Light source.
// Precomputed Environment Maps are required uniform inputs and are computed as outlined in [1].
// See our README.md on Environment Maps [3] for additional discussion.
#ifdef USE_IBL
vec3 getIBLContribution(float NdotV, vec3 N, vec3 R, int mipCount, float perceptualRoughness, vec3 diffuseColor, vec3 specularColor)
{
    float lod = (perceptualRoughness * float(mipCount));
    // retrieve a scale and bias to F0. See [1], Figure 3
    //vec3 brdf = SRGBtoLINEAR(texture2D(u_brdfLUT, vec2(NdotV, 1.0 - perceptualRoughness))).rgb;
    vec3 brdf = texture2D(u_brdfLUT, vec2(NdotV, 1.0 - perceptualRoughness)).rgb;

    vec3 diffuseLight = textureCube(u_DiffuseEnvSampler, N).rgb;

#ifdef USE_TEX_LOD
    vec3 specularLight = textureLod(u_SpecularEnvSampler, R, lod).rgb;
#else
    vec3 specularLight = textureCube(u_SpecularEnvSampler, R).rgb;
#endif

    vec3 diffuse = diffuseLight * diffuseColor;
    vec3 specular = specularLight * (specularColor * brdf.x + brdf.y);

    return diffuse + specular;
}
#endif

// Basic Lambertian diffuse
// Implementation from Lambert's Photometria https://archive.org/details/lambertsphotome00lambgoog
// See also [1], Equation 1
vec3 lambert(vec3 col)
{
    return col / M_PI;
}

vec3 Fresnel(vec3 specAlbedo, float LdotH)
{
  //return specAlbedo + (1.0 - specAlbedo) * pow((1 - LdotH), 5);
  // see http://seblagarde.wordpress.com/2012/06/03/spherical-gaussien-approximation-for-blinn-phong-phong-and-fresnel/
  // pow(1-LdotH, 5) = exp2((-5.55473 * ldotH - 6.98316) * ldotH)
  return specAlbedo + ( saturate( 50.0 * specAlbedo.g ) - specAlbedo ) * exp2( (-5.55473 * LdotH - 6.98316) * LdotH );
}


// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15

//reflectance0 = specularColor
// LdotH == VdotH
vec3 Fresnel(vec3 specularColor, float LdotH, float reflectance90)
{
  return specularColor + (reflectance90 - specularColor) * pow(saturate(1 - LdotH), 5);
}

float GGX_V1(in float m2, in float nDotX)
{
  return 1.0 / (nDotX + sqrt(m2 + (1 - m2) * nDotX * nDotX));
}

// This calculates the specular geometric attenuation (aka G()),
// where rougher material will reflect less light back to the viewer.
// This implementation is based on [1] Equation 4, and we adopt their modifications to
// alphaRoughness as input as originally proposed in [2].
float geometricOcclusion(float NdotL, float NdotV, float alphaRoughness)
{
    float m2 = alphaRoughness * alphaRoughness;

    float attenuationL = 2.0 * NdotL * GGX_V1(m2, NdotL);
    float attenuationV = 2.0 * NdotV * GGX_V1(m2, NdotV);
    return attenuationL * attenuationV;
}

// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
float D_GGX(float alphaRoughness, float NdotH)
{
    float m2 = alphaRoughness * alphaRoughness;
    float denom = NdotH * (NdotH * m2 - NdotH) + 1.0;
    return m2 / (M_PI * denom * denom);
}

void main()
{
    // Metallic and Roughness material properties are packed together
    // In glTF, these factors can be specified by fixed scalar values
    // or from a metallic-roughness map
    float perceptualRoughness = getRoughness();
    float metallness = getMetallness();
    // Roughness is authored as perceptual roughness; as is convention,
    // convert to material roughness by squaring the perceptual roughness [2].
    float alphaRoughness = perceptualRoughness * perceptualRoughness;

    // The albedo may be defined from a base texture or a flat color
#ifdef HAS_BASECOLORMAP
    vec4 baseColor = SRGBtoLINEAR(texture2D(u_BaseColorSampler, v_UV)) * u_BaseColorFactor;
#else
    vec4 baseColor = u_BaseColorFactor;
#endif

    vec3 diffuseColor = mix(baseColor.rgb, vec3(0.0), metallness);
    vec3 specularColor = mix(vec3(0.04), baseColor.rgb, metallness);

    // Compute reflectance.
    float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);

    // For typical incident reflectance range (between 4% to 100%) set the grazing reflectance to 100% for typical fresnel effect.
    // For very low reflectance range on highly diffuse objects (below 4%), incrementally reduce grazing reflecance to 0%.
    float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);

    vec3 N = getNormal();                             // normal at surface point
    vec3 P = v_Position;
    vec3 E = u_Camera;
    vec3 V = normalize(E - P);           // Vector from surface point to camera
    vec3 R = -normalize(reflect(V, N));

    vec3 L = normalize(u_LightDirection);             // Vector from surface point to light
    vec3 H = normalize(L + V);                        // Half vector between both L and V

    // we divide by NdotL in GGX_V1
    float NdotL = clamp(dot(N, L), 0.0001, 1.0);

    // dot(L, H) == dot(V, H)
    float LdotH = saturate(dot(L, H));

    vec3 F = Fresnel(specularColor, LdotH, reflectance90);
      //vec3 F = Fresnel(specularColor, LdotH);

    // we divide by NdotV in GGX_V1
    //float NdotV = abs(dot(N, V)) + 0.0001;
    float NdotV = clamp(dot(N, V), 0.0001, 1.0);
    float NdotH = clamp(dot(N, H), 0.0, 1.0);

    // Calculate the shading terms for the microfacet specular shading model
    float G = geometricOcclusion(NdotL, NdotV, alphaRoughness);
    float D = D_GGX(alphaRoughness, NdotH);

    // Calculation of analytical lighting contribution
    vec3 diffuse = lambert(diffuseColor);
    vec3 D_Vis = vec3(G * D / (4.0 * NdotL * NdotV));
    vec3 brdf = mix(diffuse, D_Vis, F);
    vec3 color = u_LightColor * brdf * NdotL;

    // Calculate lighting contribution from image based lighting source (IBL)
#ifdef USE_IBL
    vec3 Nrotated = u_IBLrotation * N;
    vec3 Rrotated = u_IBLrotation * R;
    //color += u_IBLIntensity * getIBLContribution(NdotV, N, R, mipCount, perceptualRoughness, diffuseColor, specularColor);
    color += u_IBLIntensity * getIBLContribution(NdotV, Nrotated, Rrotated, mipCount, perceptualRoughness, diffuseColor, specularColor);
#endif

#ifdef HAS_OCCLUSIONMAP
    float ao = texture2D(u_OcclusionSampler, v_UV).r;
    color = mix(color, color * ao, u_OcclusionStrength);
#endif

#ifdef HAS_EMISSIVEMAP
    vec3 emissive = SRGBtoLINEAR(texture2D(u_EmissiveSampler, v_UV)).rgb * u_EmissiveFactor;
    color += emissive;
#endif

    if (u_EnableHDR)
        FragColor = vec4(color, baseColor.a);
    else
        FragColor = vec4(pow(color,vec3(1.0/2.2)), baseColor.a);

    #ifdef USE_LINEARDEPTHBUFFER
      gl_FragDepth = length(v_Position) / u_FarClip;
    #endif
}
)";
} // namespace cs::graphics::internal

#endif // CS_GRAPHICS_PBR_FRAGMENT_SHADER_HPP
