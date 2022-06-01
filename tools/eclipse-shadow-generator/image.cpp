////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

namespace image {

void save(double* data, size_t width, size_t height, std::string const& fileName) {
  std::vector<float> floatData(data, data + width * height);
  stbi_write_hdr(fileName.c_str(), width, height, 1, floatData.data());
}

void save(float* data, size_t width, size_t height, std::string const& fileName) {
  stbi_write_hdr(fileName.c_str(), width, height, 1, data);
}

void save(glm::dvec3* data, size_t width, size_t height, std::string const& fileName) {
  std::vector<glm::vec3> floatData(data, data + width * height);
  stbi_write_hdr(fileName.c_str(), width, height, 3, &floatData[0][0]);
}

void save(glm::vec3* data, size_t width, size_t height, std::string const& fileName) {
  stbi_write_hdr(fileName.c_str(), width, height, 3, &data[0][0]);
}

} // namespace image
