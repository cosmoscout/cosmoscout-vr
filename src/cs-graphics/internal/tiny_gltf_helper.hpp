////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_TINY_GLTF_HELPER
#define CS_GRAPHICS_TINY_GLTF_HELPER
#include <GL/glew.h>
#include <tiny_gltf.h>
#include <tuple>
#include <vector>

namespace cs::graphics::internal {

/// Returns the number of components from the tinygltf::Accessor.
// constexpr int sizeFromGltfAccessorType(tinygltf::Accessor const& accessor)
inline int sizeFromGltfAccessorType(tinygltf::Accessor const& accessor) {
  if (accessor.type == TINYGLTF_TYPE_SCALAR) {
    return 1;
  }
  if (accessor.type == TINYGLTF_TYPE_VEC2) {
    return 2;
  }
  if (accessor.type == TINYGLTF_TYPE_VEC3) {
    return 3;
  }
  if (accessor.type == TINYGLTF_TYPE_VEC4) {
    return 4;
  }
  if (accessor.type == TINYGLTF_TYPE_MAT2) {
    return 4;
  }
  if (accessor.type == TINYGLTF_TYPE_MAT3) {
    return 9;
  }
  // accessor.type == tinygltf::TINYGLTF_TYPE_MAT4
  return 16;
}

/// Returns the name of the tinygltf::Accessor.
inline std::string stringFromGltfAccessorType(tinygltf::Accessor const& accessor) {
  if (accessor.type == TINYGLTF_TYPE_SCALAR) {
    return "TINYGLTF_TYPE_SCALAR";
  }
  if (accessor.type == TINYGLTF_TYPE_VEC2) {
    return "TINYGLTF_TYPE_VEC2";
  }
  if (accessor.type == TINYGLTF_TYPE_VEC3) {
    return "TINYGLTF_TYPE_VEC3";
  }
  if (accessor.type == TINYGLTF_TYPE_VEC4) {
    return "TINYGLTF_TYPE_VEC4";
  }
  if (accessor.type == TINYGLTF_TYPE_MAT2) {
    return "TINYGLTF_TYPE_Mat2";
  }
  if (accessor.type == TINYGLTF_TYPE_MAT3) {
    return "TINYGLTF_TYPE_Mat3";
  }
  if (accessor.type == TINYGLTF_TYPE_MAT4) {
    return "TINYGLTF_TYPE_Mat4";
  }
  return "UNKNOWN TINYGLTF_TYPE";
}

/// Converts GLTF primitives to OpenGL primitives.
// constexpr int toGLprimitiveMode(tinygltf::Primitive const& primitive)
inline int toGLprimitiveMode(tinygltf::Primitive const& primitive) {
  if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
    return GL_TRIANGLES;
  }
  if (primitive.mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
    return GL_TRIANGLE_STRIP;
  }
  if (primitive.mode == TINYGLTF_MODE_TRIANGLE_FAN) {
    return GL_TRIANGLE_FAN;
  }
  if (primitive.mode == TINYGLTF_MODE_POINTS) {
    return GL_POINTS;
  }
  if (primitive.mode == TINYGLTF_MODE_LINE) {
    return GL_LINES;
  }
  if (primitive.mode == TINYGLTF_MODE_LINE_LOOP) {
    return GL_LINE_LOOP;
  }
  assert(0);

  return 0;
}

/// Returns the parameter with the given name from the material.
template <typename T>
auto find_material_parameter(
    tinygltf::Material const& material, std::string const& name, T const& def) -> T {
  auto it = material.values.find(name);

  bool found = it != material.values.end();

  if (!found) {
    it    = material.additionalValues.find(name);
    found = it != material.additionalValues.end();
  }

  if (!found) {
    return def;
  }
  auto const& parameter = it->second;
  T           value{};
  for (typename T::length_type i = 0;
       i < std::min(
               value.length(), static_cast<typename T::length_type>(parameter.number_array.size()));
       ++i) {
    value[i] = static_cast<typename T::value_type>(parameter.number_array[i]);
  }
  return value;
}

/// Returns the parameter with the given name from the material.
template <>
auto find_material_parameter(
    tinygltf::Material const& material, std::string const& name, float const& def) -> float;

/// Returns the id of the texture with the given name from the material.
int find_texture_index(tinygltf::Material const& material, std::string const& name);

/// Loads a tinygltf::Image from the given filepath.
tinygltf::Image loadImage(std::string const& filepath);
} // namespace cs::graphics::internal
#endif // CS_GRAPHICS_TINY_GLTF_HELPER
