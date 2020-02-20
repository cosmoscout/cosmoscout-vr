////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for      //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gltfmodel.hpp"
#include "../../cs-utils/utils.hpp"

#include <GL/glew.h>

#include "pbr_fragment_shader.hpp"
#include "pbr_vertex_shader.hpp"
#include "stb_image_helper.hpp"
#include "tiny_gltf_helper.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaMath/VistaBoundingBox.h>
#include <algorithm>
#include <fstream>
#include <gli/gli.hpp>
#include <spdlog/spdlog.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CheckGLErrors(desc)                                                                        \
  {                                                                                                \
    GLenum e = glGetError();                                                                       \
    if (e != GL_NO_ERROR) {                                                                        \
      printf("WARNING from vista-gltf: OpenGL error in \"%s\": %d (%d) %s:%d\n", desc, e, e,       \
          __FILE__, __LINE__);                                                                     \
    }                                                                                              \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cs::graphics::internal {

////////////////////////////////////////////////////////////////////////////////////////////////////

tinygltf::Sampler defaultTinygltfSampler() {
  tinygltf::Sampler sampler;
  sampler.minFilter = GL_LINEAR;
  sampler.magFilter = GL_LINEAR;
  sampler.wrapS     = GL_REPEAT;
  sampler.wrapT     = GL_REPEAT;
  return sampler;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* vertex_shader_source = R"(
#version 330
out vec2 texCoord;
void main() {
  texCoord = vec2(float(gl_VertexID / 2) * 2.0, float(gl_VertexID % 2) * 2.0);
  gl_Position = vec4 (float(gl_VertexID / 2) * 4 - 1, float(gl_VertexID % 2) * 4 - 1, 0, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* filter_fragment_source = R"(
#version 430
#extension GL_NV_shadow_samplers_cube : enable

uniform samplerCube u_InputCubemap;
uniform int u_Level;
uniform int u_MipLevels;
uniform int u_Face;

in vec2 texCoord;

#define saturate(x) clamp(x, 0, 1)
#define PI 3.14159265359
out vec4 FragColor;

vec3 sRGB_to_linear2(vec3 c)
{
  return mix(vec3(c * (1.0 / 12.92)),
             pow((c + 0.055)/1.055, vec3(2.4)),
             greaterThanEqual(c, vec3(0.04045)));
}

float sRGB_to_linear(float c)
{
  if(c < 0.04045)
    return (c < 0.0) ? 0.0: c * (1.0 / 12.92);
  else
    return pow((c + 0.055)/1.055, 2.4);
}

vec3 sRGB_to_linear(vec3 sRGB)
{
  return vec3( sRGB_to_linear(sRGB.r),
               sRGB_to_linear(sRGB.g),
               sRGB_to_linear(sRGB.b));
}

// Brian Karis s2013_pbs_epic_notes_v2.pdf
vec3 ImportanceSampleGGX( vec2 Xi, float Roughness, vec3 N)
{
  //float a = pow(Roughness + 1, 2);
  float a = Roughness * Roughness;
  
  float Phi = 2 * PI * Xi.x;
  float CosTheta = sqrt( (1.0 - Xi.y) / ( 1.0 + (a*a - 1.0) * Xi.y ) );
  float SinTheta = sqrt( 1.0 - CosTheta * CosTheta );
  
  vec3 H = vec3(SinTheta * cos( Phi ), SinTheta * sin( Phi ), CosTheta);
  vec3 up = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);

  vec3 TangentX = normalize( cross( up, N ) );
  vec3 TangentY = cross( N, TangentX );
  
  // Tangent to world space
  return normalize(TangentX * H.x + TangentY * H.y + N * H.z);
}

// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
float radicalInverse_VdC(uint bits) 
{
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  
  return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint n)
{ 
  return vec2(float(i)/float(n), radicalInverse_VdC(i));
}

vec3 PrefilterEnvMap( float Roughness, vec3 R )
{
  vec3 N = R;
  vec3 V = R;
  vec3 PrefilteredColor = vec3(0.0);
  //const uint NumSamples = 128;
  const uint NumSamples = 1024u;
  float TotalWeight = 0.0000001;
  
  for (uint i = 0u; i < NumSamples; i++ ) {
    vec2 Xi = Hammersley( i, NumSamples );
    vec3 H = ImportanceSampleGGX( Xi, Roughness, N );
    vec3 L = 2 * dot(V, H) * H - V;
    float NoL = saturate(dot(N, L));
    if (NoL > 0) {
      //PrefilteredColor += probe_decode_hdr( t_probe_cubemap.SampleLevel( s_base, vec, mip_index ) ).rgb * NoL;
      //PrefilteredColor += textureLod(u_InputCubemap, L, u_SpecularEnvLevel)).rgb * NoL;

      PrefilteredColor += sRGB_to_linear(textureLod(u_InputCubemap, L, 0).rgb) * NoL;

      TotalWeight += NoL;
    }
  }
  return PrefilteredColor / TotalWeight;
}

ivec3 remapIndices[6] = ivec3[](
  ivec3(2,1,0),
  ivec3(2,1,0),
  ivec3(0,2,1),
  ivec3(0,2,1),
  ivec3(0,1,2),
  ivec3(0,1,2)
);

vec3 remapSign[6] = vec3[](
  vec3(  1, -1, -1),
  vec3( -1, -1,  1),
  vec3(  1,  1,  1),
  vec3(  1, -1, -1),
  vec3(  1, -1,  1),
  vec3( -1, -1, -1)
);

ivec3 Index = remapIndices[u_Face];
vec3 Sign = remapSign[u_Face];

void main() {
  float Roughness = float(u_Level) / float(u_MipLevels);

  vec3 dir = vec3(texCoord * 2.0 - 1.0, 1.0);
  vec3 R = Sign * vec3(dir[Index.x], dir[Index.y], dir[Index.z]);
  R = normalize(R);

  FragColor = vec4(PrefilterEnvMap(Roughness, R), 1.0);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* irradiance_fragment_source = R"(
#version 430
#extension GL_NV_shadow_samplers_cube : enable

uniform samplerCube u_InputCubemap;
uniform int u_Face;

#define saturate(x) clamp(x, 0, 1)
#define PI 3.14159265359

in vec2 texCoord;
out vec4 FragColor;

vec3 sRGB_to_linear(vec3 c)
{
  return mix(vec3(c * (1.0 / 12.92)),
             pow((c + 0.055)/1.055, vec3(2.4)),
             greaterThanEqual(c, vec3(0.04045)));
}

ivec3 remapIndices[6] = ivec3[](
  ivec3(2,1,0), // POSITIVE_X
  ivec3(2,1,0), // NEGATIVE_X
  ivec3(0,2,1), // POSITIVE_Y
  ivec3(0,2,1), // NEGATIVE_Y
  ivec3(0,1,2), // POSITIVE_Z
  ivec3(0,1,2)  // NEGATIVE_Z
);

vec3 remapSign[6] = vec3[](
  vec3(  1, -1, -1),
  vec3( -1, -1,  1),
  vec3(  1,  1,  1),
  vec3(  1, -1, -1),
  vec3(  1, -1,  1),
  vec3( -1, -1, -1)
);

ivec3 Index = remapIndices[u_Face];
vec3 Sign = remapSign[u_Face];

void main() {
  vec3 dir = vec3(texCoord * 2.0 - 1.0, 1.0);
  vec3 R = Sign * vec3(dir[Index.x], dir[Index.y], dir[Index.z]);
  R = normalize(R);
  vec3 N = R;

  // from https://learnopengl.com/PBR/IBL/Diffuse-irradiance
  vec3 irradiance = vec3(0.0);

  vec3 up    = vec3(0.0, 1.0, 0.0);
  vec3 right = cross(up, N);
  up            = cross(N, right);
       
  float sampleDelta = 0.025;
  float nrSamples = 0.0;

  for (float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta) {
    for (float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta) {
      // spherical to cartesian (in tangent space)
      vec3 tangentSample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
      // tangent space to world
      vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N; 

      irradiance += sRGB_to_linear(texture(u_InputCubemap, sampleVec).rgb) * cos(theta) * sin(theta);

      nrSamples++;
    }
  }
  irradiance = PI * irradiance * (1.0 / float(nrSamples));

  FragColor = vec4(irradiance, 1.0);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* compute_brdf_lut = R"(
#version 430
#extension GL_NV_shadow_samplers_cube : enable
layout (rg16f, binding = 0) writeonly uniform image2D outputImage;
layout (local_size_x = 16, local_size_y = 16) in;

#define saturate(x) clamp(x, 0, 1)
#define PI 3.14159265359

// Brian Karis s2013_pbs_epic_notes_v2.pdf
vec3 ImportanceSampleGGX( vec2 Xi, float Roughness, vec3 N)
{
  //float a = pow(Roughness + 1, 2);
  float a = Roughness * Roughness;
  
  float Phi = 2 * PI * Xi.x;
  float CosTheta = sqrt( (1.0 - Xi.y) / ( 1.0 + (a*a - 1.0) * Xi.y ) );
  float SinTheta = sqrt( 1.0 - CosTheta * CosTheta );
  
  vec3 H = vec3(SinTheta * cos( Phi ), SinTheta * sin( Phi ), CosTheta);
  vec3 up = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);

  vec3 TangentX = normalize( cross( up, N ) );
  vec3 TangentY = cross( N, TangentX );
  
  // Tangent to world space
  return normalize(TangentX * H.x + TangentY * H.y + N * H.z);
}

// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
float radicalInverse_VdC(uint bits) 
{
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  
  return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint n)
{ 
  return vec2(float(i)/float(n), radicalInverse_VdC(i));
}

// http://graphicrants.blogspot.com.au/2013/08/specular-brdf-reference.html
float GGX(float nDotV, float a) {
  // lipsryme, http://www.gamedev.net/topic/658769-ue4-ibl-glsl/
  // http://graphicrants.blogspot.com.au/2013/08/specular-brdf-reference.html
  float k = a / 2.0;
  return nDotV / (nDotV * (1.0 - k) + k);
} 

float G_Smith(float Roughness, float nDotV, float nDotL) {
  // lipsryme, http://www.gamedev.net/topic/658769-ue4-ibl-glsl/ 
  float a = Roughness * Roughness;
  return GGX(nDotL, a) * GGX(nDotV, a);
}

vec2 IntegrateBRDF( float Roughness, float NoV , vec3 N) {
    vec3 V = vec3( sqrt ( 1.0 - NoV * NoV ) //sin
                 , 0.0
                 , NoV); // cos
    float A = 0.0;
    float B = 0.0;
    const uint NumSamples = 1024u;
    for ( uint i = 0u; i < NumSamples; i++ ) {
        vec2 Xi = Hammersley( i, NumSamples );
        vec3 H = ImportanceSampleGGX( Xi, Roughness, N );
        vec3 L = 2.0 * dot(V, H) * H - V;
        float NoL = saturate(L.z);
        float NoH = saturate(H.z);
        float VoH = saturate(dot(V, H));
        if ( NoL > 0.0 ) {
            float G = G_Smith(Roughness, NoV, NoL);
            float G_Vis = G * VoH / (NoH * NoV);
            float Fc = pow(1.0 - VoH, 5.0);
            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    return vec2(A, B) / float(NumSamples);
}

void main()
{
  ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size = imageSize(outputImage);

  if (storePos.x >= size.x || storePos.y >= size.y) {
      return;
  }

  vec2 fragCoord = vec2(storePos) + vec2(0.5);
  vec2 resolution = vec2(size);
  vec2 uv = fragCoord / resolution;

  vec3 N = vec3(0,0,1); 
  float NdotV = uv.x;
  float Roughness = uv.y;

  vec2 result = IntegrateBRDF(Roughness, NdotV, N);

  imageStore(outputImage, storePos, vec4(result, 0.0, 0.0) );
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

int getProgrami(GLuint program, GLenum pname) {
  auto param = 0;
  glGetProgramiv(program, pname, &param);
  return param;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint createShader(GLenum shaderType, std::string const& srcbuf) {
  const GLchar* srcs   = srcbuf.c_str();
  auto          shader = glCreateShader(shaderType);
  glShaderSource(shader, 1, &srcs, nullptr);
  glCompileShader(shader);

  auto val = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &val);
  if (val != GL_TRUE) {
    auto log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetShaderInfoLog(shader, log_length, nullptr, v.data());
    std::string log(begin(v), end(v));
    glDeleteShader(shader);
    throw std::runtime_error(std::string("ERROR: Failed to compile shader\n") + log);
  }
  return shader;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int createProgram(const char* vs, const char* fs) {
  auto vsh = createShader(GL_VERTEX_SHADER, vs);
  auto fsh = createShader(GL_FRAGMENT_SHADER, fs);
  auto p   = glCreateProgram();
  glAttachShader(p, vsh);
  glAttachShader(p, fsh);
  glLinkProgram(p);
  glDeleteShader(vsh);
  glDeleteShader(fsh);
  return p;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint createCompute(const char* cs) {
  auto sh      = createShader(GL_COMPUTE_SHADER, cs);
  auto program = glCreateProgram();
  glAttachShader(program, sh);
  glLinkProgram(program);
  glDeleteShader(sh);

  auto rvalue = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &rvalue);
  if (!rvalue) {
    auto log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetProgramInfoLog(program, log_length, nullptr, v.data());
    std::string log(begin(v), end(v));

    throw std::runtime_error(std::string("ERROR: Failed to link compute shader\n") + log);
  }

  return program;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<GLuint> linkShader(GLuint vertShader, GLuint fragShader) {
  std::shared_ptr<GLuint> ptr(new GLuint(0), [](GLuint* ptr) {
    if (*ptr != 0u) {
      glDeleteProgram(*ptr);
    }
  });
  *ptr = glCreateProgram();

  glAttachShader(*ptr, vertShader);
  glAttachShader(*ptr, fragShader);
  glLinkProgram(*ptr);

  auto status = getProgrami(*ptr, GL_LINK_STATUS);
  assert(status == GL_TRUE && "VistaGltf.gltfmodel.linkShader: failed to link shader");

  CheckGLErrors("linkShader");
  return ptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr std::array<GLenum, 40> samplerTypes = {GL_SAMPLER_1D, GL_SAMPLER_2D, GL_SAMPLER_3D,
    GL_SAMPLER_CUBE, GL_SAMPLER_2D_RECT, GL_SAMPLER_2D_MULTISAMPLE, GL_SAMPLER_1D_ARRAY,
    GL_SAMPLER_2D_ARRAY, GL_SAMPLER_CUBE_MAP_ARRAY, GL_SAMPLER_2D_MULTISAMPLE_ARRAY,
    GL_SAMPLER_BUFFER, GL_INT_SAMPLER_1D, GL_INT_SAMPLER_2D, GL_INT_SAMPLER_3D, GL_INT_SAMPLER_CUBE,
    GL_INT_SAMPLER_2D_RECT, GL_INT_SAMPLER_2D_MULTISAMPLE, GL_INT_SAMPLER_1D_ARRAY,
    GL_INT_SAMPLER_2D_ARRAY, GL_INT_SAMPLER_CUBE_MAP_ARRAY, GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY,
    GL_INT_SAMPLER_BUFFER, GL_UNSIGNED_INT_SAMPLER_1D, GL_UNSIGNED_INT_SAMPLER_2D,
    GL_UNSIGNED_INT_SAMPLER_3D, GL_UNSIGNED_INT_SAMPLER_CUBE, GL_UNSIGNED_INT_SAMPLER_2D_RECT,
    GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE, GL_UNSIGNED_INT_SAMPLER_1D_ARRAY,
    GL_UNSIGNED_INT_SAMPLER_2D_ARRAY, GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY,
    GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY, GL_UNSIGNED_INT_SAMPLER_BUFFER,
    GL_SAMPLER_1D_SHADOW, GL_SAMPLER_2D_SHADOW, GL_SAMPLER_CUBE_SHADOW, GL_SAMPLER_2D_RECT_SHADOW,
    GL_SAMPLER_1D_ARRAY_SHADOW, GL_SAMPLER_2D_ARRAY_SHADOW, GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW};

////////////////////////////////////////////////////////////////////////////////////////////////////

bool isSamplerType(GLenum type) {
  return std::find(samplerTypes.begin(), samplerTypes.end(), type) != samplerTypes.end();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<std::map<std::string, UniformVar>, std::map<std::string, TextureVar>> get_active_uniforms(
    GLuint gl_programd) {
  auto                              n = getProgrami(gl_programd, GL_ACTIVE_UNIFORMS);
  std::map<std::string, UniformVar> uniforms;
  std::map<std::string, TextureVar> textures;
  if (n > 0) {
    std::vector<GLuint> indices(n);
    for (int i = 0; i < n; ++i) {
      auto                maxLen = getProgrami(gl_programd, GL_ACTIVE_UNIFORM_MAX_LENGTH);
      std::vector<GLchar> str(maxLen);
      auto                length = 0;
      auto                size   = 0;
      auto                type   = 0u;
      glGetActiveUniform(gl_programd, (GLuint)i, maxLen, &length, &size, &type, str.data());
      std::string name(str.data(), length);

      auto loc = glGetUniformLocation(gl_programd, name.c_str());
      if (isSamplerType(type)) {
        auto unit = static_cast<unsigned int>(textures.size());
        glUniform1i(loc, unit);
        textures[name] = TextureVar{name, loc, unit, type};
      } else {
        uniforms[name] = UniformVar{name, loc, size, type};
      }
    }
  }
  CheckGLErrors("get_active_uniforms");
  return {uniforms, textures};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLProgramInfo getProgramInfo(unsigned int program) {
  GLProgramInfo info;
  glUseProgram(program);

  auto count = getProgrami(program, GL_ACTIVE_ATTRIBUTES);
  // glGetActiveAttrib(program, index, maxLength, size, type, name)
  std::map<std::string, std::string> attributeMap{{"a_Position", "POSITION"},
      {"a_Normal", "NORMAL"}, {"a_Tangent", "TANGENT"}, {"a_UV", "TEXCOORD_0"}};
  for (auto const& mapping : attributeMap) {
    auto loc = glGetAttribLocation(program, mapping.first.c_str());
    if (loc != -1) {
      info.pbr_attributes[mapping.second] = loc;
    }
  }

  info.u_MVPMatrix_loc    = glGetUniformLocation(program, "u_MVPMatrix");
  info.u_ModelMatrix_loc  = glGetUniformLocation(program, "u_ModelMatrix");
  info.u_NormalMatrix_loc = glGetUniformLocation(program, "u_NormalMatrix");

  info.u_LightDirection_loc = glGetUniformLocation(program, "u_LightDirection");
  info.u_LightColor_loc     = glGetUniformLocation(program, "u_LightColor");

  info.u_DiffuseEnvSampler_loc  = glGetUniformLocation(program, "u_DiffuseEnvSampler");
  info.u_SpecularEnvSampler_loc = glGetUniformLocation(program, "u_SpecularEnvSampler");
  info.u_brdfLUT_loc            = glGetUniformLocation(program, "u_brdfLUT");
  info.u_IBLIntensity_loc       = glGetUniformLocation(program, "u_IBLIntensity");
  info.u_IBLrotation_loc        = glGetUniformLocation(program, "u_IBLrotation");

  info.u_NormalScale_loc       = glGetUniformLocation(program, "u_NormalScale");
  info.u_EmissiveFactor_loc    = glGetUniformLocation(program, "u_EmissiveFactor");
  info.u_OcclusionStrength_loc = glGetUniformLocation(program, "u_OcclusionStrength");

  info.u_MetallicRoughnessValues_loc = glGetUniformLocation(program, "u_MetallicRoughnessValues");
  info.u_BaseColorFactor_loc         = glGetUniformLocation(program, "u_BaseColorFactor");
  info.u_Camera_loc                  = glGetUniformLocation(program, "u_Camera");
  info.u_FarClip_loc                 = glGetUniformLocation(program, "u_FarClip");

  auto usTexs   = get_active_uniforms(program);
  info.uniforms = usTexs.first;
  info.textures = usTexs.second;

  glUseProgram(0);

  CheckGLErrors("getProgramInfo");
  return info;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Buffer getOrCreateBufferObject(std::map<int, Buffer>& bufferMap, tinygltf::Model const& gltf,
    unsigned int bufferViewIndex, unsigned int target) {
  auto it = bufferMap.find(bufferViewIndex);
  if (it != bufferMap.end()) {
    if (it->second.target != target) {
      spdlog::warn(
          "Failed to create GLTF BufferObject: Target is different from Buffer.target for {}!",
          bufferViewIndex);
    }
    return it->second;
  } // else create

  const tinygltf::BufferView& bufferView = gltf.bufferViews[bufferViewIndex];
  const tinygltf::Buffer&     buffer     = gltf.buffers[bufferView.buffer];

  auto ptr = std::shared_ptr<GLuint>(new GLuint(0), [](GLuint* ptr) {
    if (*ptr != 0u) {
      glDeleteBuffers(1, ptr);
    }
  });

  glGenBuffers(1, ptr.get());
  glBindBuffer(target, *ptr);
  glBufferData(
      target, bufferView.byteLength, &buffer.data.at(0) + bufferView.byteOffset, GL_STATIC_DRAW);
  glBindBuffer(target, 0);
  bufferMap[bufferViewIndex] = Buffer{target, ptr};
  CheckGLErrors("getOrCreateBufferObject");
  return {target, ptr};
}

Texture uploadCubemap(gli::texture_cube const& gliTex) {
  gli::gl GL(gli::gl::PROFILE_GL33);
  auto    format = GL.translate(gliTex.format(), gliTex.swizzles());
  GLenum  Target = GL.translate(gliTex.target());

  std::shared_ptr<GLuint> ptr(new GLuint(0), [](GLuint* ptr) {
    if (*ptr != 0u) {
      glDeleteTextures(1, ptr);
    }
  });
  glGenTextures(1, ptr.get());
  glBindTexture(GL_TEXTURE_CUBE_MAP, *ptr);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

  glTexParameteri(Target, GL_TEXTURE_BASE_LEVEL, 0);
  glTexParameteri(Target, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(gliTex.levels() - 1));
  glTexParameteri(Target, GL_TEXTURE_SWIZZLE_R, format.Swizzles[0]);
  glTexParameteri(Target, GL_TEXTURE_SWIZZLE_G, format.Swizzles[1]);
  glTexParameteri(Target, GL_TEXTURE_SWIZZLE_B, format.Swizzles[2]);
  glTexParameteri(Target, GL_TEXTURE_SWIZZLE_A, format.Swizzles[3]);

  glTexStorage2D(Target, static_cast<GLint>(gliTex.levels()), format.Internal, gliTex.extent().x,
      gliTex.extent().y);

  std::size_t layer = 0;
  for (std::size_t level = 0; level < gliTex.levels(); ++level) {
    for (std::size_t face = 0; face < gliTex.faces(); ++face) {
      auto target = static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face);
      auto extent = gliTex.extent(level);
      if (gli::is_compressed(gliTex.format())) {
        glCompressedTexSubImage2D(target, static_cast<GLint>(level), 0, 0, extent.x, extent.y,
            format.Internal, static_cast<GLsizei>(gliTex.size(level)),
            gliTex.data(layer, face, level));
      } else {
        glTexSubImage2D(target, static_cast<GLint>(level), 0, 0, extent.x, extent.y,
            format.External, format.Type, gliTex.data(layer, face, level));
      }
    }
  }

  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
  CheckGLErrors("uploadCubemap");

  std::shared_ptr<GLuint> specularEnvMapSampler(new GLuint(0), [](GLuint* ptr) {
    if (*ptr != 0u) {
      glDeleteSamplers(1, ptr);
    }
  });
  glGenSamplers(1, specularEnvMapSampler.get());
  glSamplerParameteri(*specularEnvMapSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  //*specularEnvMapSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glSamplerParameteri(*specularEnvMapSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glSamplerParameteri(*specularEnvMapSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glSamplerParameteri(*specularEnvMapSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glSamplerParameteri(*specularEnvMapSampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

  return Texture{GL_TEXTURE_CUBE_MAP, specularEnvMapSampler, ptr};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gli::texture_cube prefilterCubemapGGX(gli::texture_cube const& inputCubemap, std::size_t levels) {
  auto vao = 0u;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  auto program = createProgram(vertex_shader_source, filter_fragment_source);
  glUseProgram((GLuint)program);
  CheckGLErrors("after glUseProgram");
  auto info        = get_active_uniforms((GLuint)program);
  auto uniformVars = info.first;
  auto textureVars = info.second;

  auto inputCubemapTex = uploadCubemap(inputCubemap);

  auto formatInternal = gli::gl::internal_format::INTERNAL_RGB16F;
  auto formatExternal = gli::gl::external_format::EXTERNAL_RGB;
  auto formatType     = gli::gl::type_format::TYPE_F16;

  gli::gl GL(gli::gl::PROFILE_GL33);
  auto    gliFormat = GL.find(formatInternal, formatExternal, formatType);

  auto width  = inputCubemap.extent().x;
  auto height = inputCubemap.extent().y;

  gli::texture_cube filteredGliTex(gliFormat, glm::ivec2(width, height), levels);

  auto it = textureVars.find("u_InputCubemap");
  if (it != textureVars.end()) {

    auto fbo = 0u;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    auto outputCubemapTex = uploadCubemap(filteredGliTex);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        *outputCubemapTex.image, 0);
    GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, drawBuffers);

    if (GL_FRAMEBUFFER_COMPLETE != glCheckFramebufferStatus(GL_FRAMEBUFFER)) {
      spdlog::error("Failed to filter GLTF cubemap: Invalid FBO!");
    }

    auto inputCubemapTexVar = it->second;
    glActiveTexture(GL_TEXTURE0 + inputCubemapTexVar.unit);
    glBindTexture(inputCubemapTex.target, *inputCubemapTex.image);
    glBindSampler(inputCubemapTexVar.unit, *inputCubemapTex.sampler);

    auto mipLevelsIterator = uniformVars.find("u_MipLevels");
    if (mipLevelsIterator != uniformVars.end()) {
      glUniform1i(mipLevelsIterator->second.location, (GLint)levels);
    }

    for (std::size_t level = 0; level < levels; ++level) {
      auto extent         = filteredGliTex.extent(level);
      auto uLevelIterator = uniformVars.find("u_Level");
      if (uLevelIterator != uniformVars.end()) {
        glUniform1i(uLevelIterator->second.location, (GLint)level);
      }

      for (std::size_t face = 0; face < 6; ++face) {
        auto target = static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, target, *outputCubemapTex.image, (GLint)level);

        uLevelIterator = uniformVars.find("u_Face");
        if (uLevelIterator != uniformVars.end()) {
          glUniform1i(uLevelIterator->second.location, (GLint)face);
        }

        glViewport(0, 0, extent.x, extent.y);
        glScissor(0, 0, extent.x, extent.y);
        glDrawArrays(GL_TRIANGLES, 0, 3);
      }
    }

    glActiveTexture(GL_TEXTURE0 + inputCubemapTexVar.unit);
    glBindTexture(inputCubemapTex.target, 0);
    glBindSampler(inputCubemapTexVar.unit, 0);

    glUseProgram(0);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);

    // Fetch cubemap
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(outputCubemapTex.target, *outputCubemapTex.image);
    for (std::size_t level = 0; level < filteredGliTex.levels(); ++level) {
      for (std::size_t face = 0; face < filteredGliTex.faces(); ++face) {
        auto target = static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face);
        glGetTexImage(target, (GLint)level, formatExternal,
            formatType, // GL_FLOAT,
            filteredGliTex[face][level].data());
        CheckGLErrors("after glGetTexImage");
      }
    }
    glBindTexture(outputCubemapTex.target, 0);
  }
  glDeleteVertexArrays(1, &vao);
  glDeleteProgram((GLuint)program);
  return filteredGliTex;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gli::texture_cube irradianceCubemap(gli::texture_cube const& inputCubemap, int width, int height) {
  GLuint vao = 0;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  auto program = createProgram(vertex_shader_source, irradiance_fragment_source);
  glUseProgram((GLuint)program);
  CheckGLErrors("after glUseProgram");
  auto info        = get_active_uniforms((GLuint)program);
  auto uniformVars = info.first;
  auto textureVars = info.second;

  auto inputCubemapTex = uploadCubemap(inputCubemap);

  auto formatInternal = gli::gl::internal_format::INTERNAL_RGB16F;
  auto formatExternal = gli::gl::external_format::EXTERNAL_RGB;
  auto formatType     = gli::gl::type_format::TYPE_F16;

  gli::gl GL(gli::gl::PROFILE_GL33);
  auto    gliFormat = GL.find(formatInternal, formatExternal, formatType);

  gli::texture_cube filteredGliTex(gliFormat, glm::ivec2(width, height), 1);

  auto it = textureVars.find("u_InputCubemap");
  if (it != textureVars.end()) {
    auto fbo = 0u;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    auto outputCubemapTex = uploadCubemap(filteredGliTex);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        *outputCubemapTex.image, 0);
    GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, drawBuffers);

    if (GL_FRAMEBUFFER_COMPLETE != glCheckFramebufferStatus(GL_FRAMEBUFFER)) {
      spdlog::error("Failed to filter GLTF cubemap: Invalid FBO!");
    }

    auto inputCubemapTexVar = it->second;
    glActiveTexture(GL_TEXTURE0 + inputCubemapTexVar.unit);
    glBindTexture(inputCubemapTex.target, *inputCubemapTex.image);
    glBindSampler(inputCubemapTexVar.unit, *inputCubemapTex.sampler);

    auto level = 0;
    for (auto face = 0u; face < 6u; ++face) {
      auto target = GL_TEXTURE_CUBE_MAP_POSITIVE_X + face;
      glFramebufferTexture2D(
          GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, target, *outputCubemapTex.image, level);

      auto faceIterator = uniformVars.find("u_Face");
      if (faceIterator != uniformVars.end()) {
        glUniform1i(faceIterator->second.location, face);
      }

      glViewport(0, 0, width, height);
      glScissor(0, 0, width, height);
      glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    glActiveTexture(GL_TEXTURE0 + inputCubemapTexVar.unit);
    glBindTexture(inputCubemapTex.target, 0);
    glBindSampler(inputCubemapTexVar.unit, 0);

    glUseProgram(0);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);

    // Fetch cubemap
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(outputCubemapTex.target, *outputCubemapTex.image);
    for (auto level = 0u; level < filteredGliTex.levels(); ++level) {
      for (auto face = 0u; face < filteredGliTex.faces(); ++face) {
        auto target = static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face);
        glGetTexImage(target, level, formatExternal,
            formatType, // GL_FLOAT,
            filteredGliTex[face][level].data());
        CheckGLErrors("after glGetTexImage");
      }
    }
    glBindTexture(outputCubemapTex.target, 0);
  }
  glDeleteVertexArrays(1, &vao);
  glDeleteProgram((GLuint)program);
  return filteredGliTex;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<GLuint> createGPUimage(tinygltf::Image const& img, bool withMipmaps) {
  std::shared_ptr<GLuint> ptr(new GLuint(0), [](GLuint* ptr) {
    if (*ptr != 0u)
      glDeleteTextures(1, ptr);
  });
  glGenTextures(1, ptr.get());
  glBindTexture(GL_TEXTURE_2D, *ptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  auto internalFormat = stbi_component_to_internal_format(img.component);
  auto format         = stbi_component_to_format(img.component);

  glTexImage2D(GL_TEXTURE_2D, // target
      0,                      // level
      internalFormat,
      img.width,  // width
      img.height, // height
      0,          // border
      format, GL_UNSIGNED_BYTE,
      img.image.data()); // data
  if (withMipmaps) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  CheckGLErrors("createGPUimage");
  return ptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<GLuint> createGPUsampler(tinygltf::Sampler const& s) {
  std::shared_ptr<GLuint> ptr(new GLuint(0), [](GLuint* ptr) {
    if (*ptr != 0u) {
      glDeleteSamplers(1, ptr);
    }
  });
  glGenSamplers(1, ptr.get());
  glSamplerParameteri(*ptr, GL_TEXTURE_MIN_FILTER, s.minFilter);
  glSamplerParameteri(*ptr, GL_TEXTURE_MAG_FILTER, s.magFilter);
  glSamplerParameteri(*ptr, GL_TEXTURE_WRAP_S, s.wrapS);
  glSamplerParameteri(*ptr, GL_TEXTURE_WRAP_T, s.wrapT);
  CheckGLErrors("createGPUsampler");
  return ptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Texture createBrdfLUT(int width, int height) {
  std::shared_ptr<GLuint> texture_ptr(new GLuint(0), [](GLuint* ptr) {
    if (*ptr != 0u) {
      glDeleteTextures(1, ptr);
    }
  });
  glGenTextures(1, texture_ptr.get());
  glBindTexture(GL_TEXTURE_2D, *texture_ptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, width, height, 0, GL_RG, GL_HALF_FLOAT, nullptr);

  auto program = createCompute(compute_brdf_lut);

  glUseProgram(program);
  glBindImageTexture(0, *texture_ptr, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);
  glDispatchCompute((GLuint)width / 16, (GLuint)height / 16, 1);
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
  glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);

  glDeleteProgram(program);
  CheckGLErrors("in createBrdfLUT");

  tinygltf::Sampler sampler;
  sampler.name      = "brdfLUT";
  sampler.minFilter = GL_LINEAR;
  sampler.magFilter = GL_LINEAR;
  sampler.wrapS     = GL_CLAMP_TO_EDGE;
  sampler.wrapT     = GL_CLAMP_TO_EDGE;

  return Texture{GL_TEXTURE_2D, createGPUsampler(sampler), texture_ptr};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GltfShared::buildMeshes(tinygltf::Model const& gltf) {
  for (auto const& gltfMesh : gltf.meshes) {
    Mesh mesh;
    for (auto const& primitive : gltfMesh.primitives) {
      auto it = primitive.attributes.find("POSITION");

      if (it != primitive.attributes.end()) {
        auto& a = gltf.accessors[it->second];

        if (a.minValues.size() == 3) {
          mesh.minPos[0] = std::min(mesh.minPos[0], float(a.minValues[0]));
          mesh.minPos[1] = std::min(mesh.minPos[1], float(a.minValues[1]));
          mesh.minPos[2] = std::min(mesh.minPos[2], float(a.minValues[2]));
        }

        if (a.maxValues.size() == 3) {
          mesh.maxPos[0] = std::max(mesh.maxPos[0], float(a.maxValues[0]));
          mesh.maxPos[1] = std::max(mesh.maxPos[1], float(a.maxValues[1]));
          mesh.maxPos[2] = std::max(mesh.maxPos[2], float(a.maxValues[2]));
        }
      }
      mesh.primitives.push_back(createMeshPrimitive(gltf, primitive));
    }
    meshes.push_back(mesh);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Primitive GltfShared::createMeshPrimitive(
    tinygltf::Model const& gltf, tinygltf::Primitive const& primitive) {
  Primitive myPrimitive;
  myPrimitive.hasIndices = primitive.indices >= 0;

  std::string definesVS = "#version 330\n";
  std::string definesFS = "#version 330\n";
  // = "#version 450\n";
  definesFS += "#define USE_IBL\n#define USE_TEX_LOD\n";

  if (primitive.attributes.count("NORMAL")) {
    definesVS += "#define HAS_NORMALS\n";
    definesFS += "#define HAS_NORMALS\n";
  }

  if (primitive.attributes.count("TEXCOORD_0")) {
    definesVS += "#define HAS_UV\n";
  }

  if (primitive.attributes.count("TANGENT")) {
    definesVS += "#define HAS_TANGENTS\n";
    definesFS += "#define HAS_TANGENTS\n";
  }

  if (m_linearDepthBuffer) {
    definesFS += "#define USE_LINEARDEPTHBUFFER\n";
  }

  tinygltf::Material const* material = nullptr;

  if (primitive.material >= 0) {
    material = &gltf.materials[primitive.material];
  }

  if (material) {
    if (material->values.count("baseColorTexture")) {
      definesFS += "#define HAS_BASECOLORMAP\n";
    }

    if (material->values.count("metallicRoughnessTexture")) {
      definesFS += "#define HAS_METALROUGHNESSMAP\n";
    }

    if (material->additionalValues.count("normalTexture")) {
      definesFS += "#define HAS_NORMALMAP\n";
    }

    if (material->additionalValues.count("emissiveTexture")) {
      definesFS += "#define HAS_EMISSIVEMAP\n";
    }

    if (material->additionalValues.count("occlusionTexture")) {
      definesFS += "#define HAS_OCCLUSIONMAP\n";
    }
  }

  auto vertSource = definesVS + GLTF_VERT;
  auto fragSource = definesFS + GLTF_FRAG;

  auto vertId            = createShader(GL_VERTEX_SHADER, vertSource);
  auto fragId            = createShader(GL_FRAGMENT_SHADER, fragSource);
  myPrimitive.programPtr = linkShader(vertId, fragId);
  glDeleteShader(vertId);
  glDeleteShader(fragId);
  myPrimitive.programInfo = getProgramInfo(*myPrimitive.programPtr);

  if (material) {
    myPrimitive.baseColorFactor =
        find_material_parameter(*material, "baseColorFactor", glm::vec4(1.0f));
    myPrimitive.metallicRoughnessValues.x =
        find_material_parameter(*material, "metallicFactor", 1.0f);
    myPrimitive.metallicRoughnessValues.y =
        find_material_parameter(*material, "roughnessFactor", 1.0f);
    myPrimitive.emissiveFactor =
        find_material_parameter(*material, "emissiveFactor", glm::vec3(0.0f));

    std::map<std::string, std::string> materialTextures{{"baseColorTexture", "u_BaseColorSampler"},
        {"metallicRoughnessTexture", "u_MetallicRoughnessSampler"},
        {"occlusionTexture", "u_OcclusionSampler"}, {"normalTexture", "u_NormalSampler"},
        {"emissiveTexture", "u_EmissiveSampler"}};
    for (auto const& pair : materialTextures) {
      int  maybeIndex = find_texture_index(*material, pair.first);
      auto texVarIter = myPrimitive.programInfo.textures.find(pair.second);
      if (maybeIndex >= 0 && texVarIter != myPrimitive.programInfo.textures.end()) {
        myPrimitive.textures.emplace_back(mextures[maybeIndex], texVarIter->second);
      }
    }
  }

  // textures used for Image Based Lighting (IBL)
  auto texVarIter = myPrimitive.programInfo.textures.find("u_brdfLUT");
  if (texVarIter != myPrimitive.programInfo.textures.end()) {
    myPrimitive.textures.emplace_back(mextures[mrdfLUTindex], texVarIter->second);
  }

  texVarIter = myPrimitive.programInfo.textures.find("u_DiffuseEnvSampler");
  if (texVarIter != myPrimitive.programInfo.textures.end()) {
    myPrimitive.textures.emplace_back(mextures[miffuseEnvMapIndex], texVarIter->second);
  }

  texVarIter = myPrimitive.programInfo.textures.find("u_SpecularEnvSampler");
  if (texVarIter != myPrimitive.programInfo.textures.end()) {
    myPrimitive.textures.emplace_back(mextures[mpecularEnvMapIndex], texVarIter->second);
  }

  // ----------------------------------------------
  // setup vertex data

  myPrimitive.vaoPtr = std::shared_ptr<GLuint>(new GLuint(0), [](GLuint* ptr) {
    if (*ptr != 0u) {
      glDeleteVertexArrays(1, ptr);
    }
  });
  glGenVertexArrays(1, myPrimitive.vaoPtr.get());
  glBindVertexArray(*myPrimitive.vaoPtr);

  // Assume TEXTURE_2D target for the texture object.
  std::map<int, Buffer> bufferMap;
  for (auto const& pair : primitive.attributes) {
    auto const&               attrName = pair.first;
    tinygltf::Accessor const& accessor = gltf.accessors[pair.second];

    auto buffer = getOrCreateBufferObject(
        bufferMap, gltf, (unsigned int)accessor.bufferView, GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, *buffer.id);
    int size = sizeFromGltfAccessorType(accessor);
    // pair.first would be "POSITION", "NORMAL", "TEXCOORD_0", ...
    auto it = myPrimitive.programInfo.pbr_attributes.find(attrName);

    if (it != myPrimitive.programInfo.pbr_attributes.end() && it->second >= 0) {
      glEnableVertexAttribArray((GLuint)it->second);
      glVertexAttribPointer((GLuint)it->second, size, (GLenum)accessor.componentType,
          GLboolean(accessor.normalized ? GL_TRUE : GL_FALSE),
          (GLsizei)gltf.bufferViews[accessor.bufferView].byteStride,
          static_cast<char*>(nullptr) + accessor.byteOffset);
      myPrimitive.verticesCount = accessor.count;
    }
  }

  if (myPrimitive.hasIndices) {
    tinygltf::Accessor const& indexAccessor = gltf.accessors[primitive.indices];
    // Import glBindBuffer(GL_ELEMENT_ARRAY_BUFFER has to be called after
    // glBindVertexArray
    auto buffer = getOrCreateBufferObject(
        bufferMap, gltf, (unsigned int)indexAccessor.bufferView, GL_ELEMENT_ARRAY_BUFFER);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *buffer.id);
    myPrimitive.indicesCount = indexAccessor.count,
    myPrimitive.indicesType  = indexAccessor.componentType,
    myPrimitive.byteOffset   = indexAccessor.byteOffset;
  }

  // done recording VAO
  glBindVertexArray(0);

  for (auto const& pair : primitive.attributes) {
    auto it = myPrimitive.programInfo.pbr_attributes.find(pair.first);
    if (it != myPrimitive.programInfo.pbr_attributes.end() && it->second >= 0) {
      glDisableVertexAttribArray((GLuint)it->second);
    }
  }

  CheckGLErrors("createMeshPrimitive");
  return myPrimitive;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hierarchically draw nodes
void Primitive::draw(glm::mat4 const& projMat, glm::mat4 const& viewMat, glm::mat4 const& modelMat,
    GltfShared const& shared) const {
  if (programPtr) {
    glUseProgram(*programPtr);
  }

  if (shared.m_linearDepthBuffer) {
    glUniform1f(programInfo.u_FarClip_loc, utils::getCurrentFarClipDistance());
  }

  auto viewMatInverse = glm::inverse(viewMat);
  auto eye            = glm::vec3(viewMatInverse[3]);
  auto normalMat      = glm::inverse(glm::transpose(glm::mat3(modelMat)));
  auto mvp            = projMat * viewMat * modelMat;
  glUniformMatrix4fv(programInfo.u_ModelMatrix_loc, 1, GL_FALSE, glm::value_ptr(modelMat));
  glUniformMatrix3fv(programInfo.u_NormalMatrix_loc, 1, GL_FALSE, glm::value_ptr(normalMat));

  glUniformMatrix4fv(programInfo.u_MVPMatrix_loc, 1, GL_FALSE, glm::value_ptr(mvp));
  glUniform3fv(programInfo.u_LightDirection_loc, 1, glm::value_ptr(shared.m_lightDirection));
  glUniform3fv(programInfo.u_LightColor_loc, 1,
      glm::value_ptr(shared.m_lightColor * shared.m_lightIntensity));
  glUniform3fv(programInfo.u_Camera_loc, 1, glm::value_ptr(eye));

  glUniform2fv(
      programInfo.u_MetallicRoughnessValues_loc, 1, glm::value_ptr(metallicRoughnessValues));
  glUniform4fv(programInfo.u_BaseColorFactor_loc, 1, glm::value_ptr(baseColorFactor));
  glUniform3fv(programInfo.u_EmissiveFactor_loc, 1, glm::value_ptr(emissiveFactor));
  glUniform1f(programInfo.u_OcclusionStrength_loc, 1.0);

  glUniform1f(programInfo.u_IBLIntensity_loc, shared.m_IBLIntensity);
  glUniformMatrix3fv(
      programInfo.u_IBLrotation_loc, 1, GL_FALSE, glm::value_ptr(shared.m_IBLrotation));

  // for (auto const & [ tex, texVar ] : textures) {
  for (auto const& pair : textures) {
    auto const& tex    = pair.first;
    auto const& texVar = pair.second;
    glActiveTexture(GL_TEXTURE0 + texVar.unit);
    glBindTexture(tex.target, *tex.image);
    glBindSampler(texVar.unit, *tex.sampler);
  }

  if (vaoPtr) {
    glBindVertexArray(*vaoPtr);
    if (hasIndices) {
      glDrawElements((GLenum)mode, (GLsizei)indicesCount, (GLenum)indicesType,
          static_cast<char*>(nullptr) + byteOffset);
    } else {
      glDrawArrays((GLenum)mode, 0, (GLsizei)verticesCount);
    }
    glBindVertexArray(0);
  }

  for (auto const& pair : textures) {
    glActiveTexture(GL_TEXTURE0 + pair.second.unit);
    glBindTexture(pair.first.target, 0);
    glBindSampler(pair.second.unit, 0);
  }

  glUseProgram(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GltfShared::init(tinygltf::Model const& gltf, std::string const& cubemapFilepath) {

  // save current viewport
  GLint current_viewport[4];
  glGetIntegerv(GL_VIEWPORT, current_viewport);

  {
    std::ifstream f(cubemapFilepath.c_str());
    if (!f.good()) {
      throw std::runtime_error("GltfShared: Cannot open cubemap: " + cubemapFilepath);
    }
  }
  std::vector<std::shared_ptr<unsigned int>> sharedImages;
  for (auto const& i : gltf.images) {
    sharedImages.emplace_back(createGPUimage(i, true));
  }
  std::vector<std::shared_ptr<unsigned int>> sharedSamplers;
  for (auto const& s : gltf.samplers) {
    sharedSamplers.emplace_back(createGPUsampler(s));
  }

  auto defaultSampler = createGPUsampler(defaultTinygltfSampler());
  for (auto const& t : gltf.textures) {
    std::shared_ptr<GLuint> sampler;
    if (t.sampler >= 0)
      sampler = sharedSamplers[t.sampler];
    else
      sampler = defaultSampler;

    mextures.emplace_back(Texture{GL_TEXTURE_2D, sampler, sharedImages.at(t.source)});
  }

  mrdfLUTindex = (int)mextures.size();
  mextures.push_back(createBrdfLUT(512, 512));

  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
  gli::texture_cube inputGliTex(gli::load(cubemapFilepath));

  // diffuse env map
  miffuseEnvMapIndex = (int)mextures.size();
  auto diffuseGliTex = irradianceCubemap(inputGliTex, 32, 32);
  mextures.push_back(uploadCubemap(diffuseGliTex));

  // specular env map
  mpecularEnvMapIndex = (int)mextures.size();
  auto specularGliTex = prefilterCubemapGGX(inputGliTex, 10);
  mextures.push_back(uploadCubemap(specularGliTex));

  buildMeshes(gltf);

  // reset current viewport
  glViewport(current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);
  glScissor(current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Mesh::draw(glm::mat4 const& projMat, glm::mat4 const& viewMat, glm::mat4 const& modelMat,
    GltfShared const& shared) const {
  for (auto const& p : primitives) {
    p.draw(projMat, viewMat, modelMat, shared);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaGltfNode::VistaGltfNode(tinygltf::Node const& node, std::shared_ptr<GltfShared> const& shared)
    : mShared(shared)
    , mName(node.name)
    , mMeshIndex(node.mesh) {
}

VistaGltfNode::~VistaGltfNode() = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

bool VistaGltfNode::Do() {
  auto renderInfo = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo();

  GLfloat glMat[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, &glMat[0]);
  glm::mat4 modelViewMat = glm::make_mat4(glMat); // == viewMat * modelMat

  glGetFloatv(GL_PROJECTION_MATRIX, &glMat[0]);
  glm::mat4 projMat  = glm::make_mat4(glMat);
  glm::mat4 viewMat  = glm::make_mat4(renderInfo->m_matCameraTransform.GetData());
  glm::mat4 modelMat = glm::inverse(viewMat) * modelViewMat;

  if (mMeshIndex >= 0 && mShared) {
    mShared->meshes[mMeshIndex].draw(projMat, viewMat, modelMat, *mShared);
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool VistaGltfNode::GetBoundingBox(VistaBoundingBox& bb) {
  if (mMeshIndex >= 0 && mShared) {
    auto& mi = mShared->meshes[mMeshIndex].minPos;
    auto& ma = mShared->meshes[mMeshIndex].maxPos;
    bb.SetBounds(glm::value_ptr(mi), glm::value_ptr(ma));
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics::internal
