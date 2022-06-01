////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <string>
#include <vector>

#include "math.cuh"

namespace image {

// Writes the given data to a hdr file. This is pretty lossy (re format)
void save(double* data, size_t width, size_t height, std::string const& fileName);
void save(float* data, size_t width, size_t height, std::string const& fileName);

// Writes the given data to a hdr file. This is pretty lossy (rgbe format)
void save(glm::dvec3* data, size_t width, size_t height, std::string const& fileName);
void save(glm::vec3* data, size_t width, size_t height, std::string const& fileName);

} // namespace image

#endif // IMAGE_HPP