////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_PBR_VERTEX_SHADER_HPP
#define CS_GRAPHICS_PBR_VERTEX_SHADER_HPP

namespace cs::graphics::internal {

const char* GLTF_VERT = R"(
in vec4 a_Position;
#ifdef HAS_NORMALS
in vec3 a_Normal;
#endif
#ifdef HAS_TANGENTS
in vec4 a_Tangent;
#endif
#ifdef HAS_UV
in vec2 a_UV;
#endif

uniform mat4 u_MVPMatrix;

uniform mat4 u_ModelMatrix;
//uniform mat4 u_ViewMatrix;
//uniform mat4 u_ProjectionMatrix;
uniform mat3 u_NormalMatrix;

out vec3 v_Position;
out vec2 v_UV;

#ifdef HAS_NORMALS
#ifdef HAS_TANGENTS
out mat3 v_TBN;
#else
out vec3 v_Normal;
#endif
#endif

void main()
{
  vec4 pos = u_ModelMatrix * a_Position;
  v_Position = vec3(pos.xyz) / pos.w;

  #ifdef HAS_NORMALS
  #ifdef HAS_TANGENTS
  vec3 normalW = normalize(u_NormalMatrix * a_Normal);
  vec3 tangentW = normalize(u_NormalMatrix * a_Tangent.xyz);
  vec3 bitangentW = cross(normalW, tangentW) * a_Tangent.w;
  v_TBN = mat3(tangentW, bitangentW, normalW);
  #else // HAS_TANGENTS != 1
  v_Normal = normalize(u_NormalMatrix * a_Normal);
  #endif
  #endif

  #ifdef HAS_UV
  v_UV = a_UV;
  #else
  v_UV = vec2(0.0);
  #endif

  gl_Position = u_MVPMatrix * a_Position; // needs w for proper perspective correction
}

)";
} // namespace cs::graphics::internal
#endif // CS_GRAPHICS_PBR_VERTEX_SHADER_HPP
