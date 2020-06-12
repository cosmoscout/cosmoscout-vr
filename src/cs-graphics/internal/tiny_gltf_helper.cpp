////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "tiny_gltf_helper.hpp"

#include <memory>
#include <stb_image.h>
#include <stdexcept>

namespace cs::graphics::internal {

template <>
auto find_material_parameter(
    tinygltf::Material const& material, std::string const& name, float const& def) -> float {
  auto it    = material.values.find(name);
  bool found = it != material.values.end();

  if (!found) {
    it    = material.additionalValues.find(name);
    found = it != material.additionalValues.end();
  }

  if (found && it->second.has_number_value) {
    return static_cast<float>(it->second.number_value); // array.front();
  }

  return def;
}

int find_texture_index(tinygltf::Material const& material, std::string const& name) {
  auto it    = material.values.find(name);
  bool found = it != material.values.end();

  if (!found) {
    it    = material.additionalValues.find(name);
    found = it != material.additionalValues.end();
  }

#if TODO_MAYBE_REMOVE
  if (!found) {
    it    = material.extCommonValues.find(name);
    found = it != material.extCommonValues.end();
  }

  if (!found) {
    it    = material.extPBRValues.find(name);
    found = it != material.extPBRValues.end();
  }
#endif

  if (found) {
    auto jIt = it->second.json_double_value.find("index");

    if (jIt != it->second.json_double_value.end()) {
      return static_cast<int>(jIt->second);
    }

    return -1;
  }

  return -1;
}

tinygltf::Image loadImage(std::string const& filepath) {
  tinygltf::Image img;

  std::unique_ptr<unsigned char> data(
      stbi_load(filepath.c_str(), &img.width, &img.height, &img.component, 0));
  if (data != nullptr) {
    img.image.resize(img.width * img.height * img.component);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::copy(data.get(), data.get() + img.width * img.height * img.component, img.image.begin());
  } else {
    throw std::runtime_error(std::string("loadImage: Unable to load ") + filepath);
  }
  return img;
}
} // namespace cs::graphics::internal
