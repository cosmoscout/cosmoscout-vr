////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GRAPHICS_OBJLOADER_HPP
#define CS_GRAPHICS_OBJLOADER_HPP

#include "cs_graphics_export.hpp"

#include <string>
#include <vector>
#include <memory>

namespace cs::graphics {

// Can be used to load .obj files and stores the data for later use.
class CS_GRAPHICS_EXPORT ObjLoader {
 public:
  ObjLoader(const std::string& objFilePath);

  ObjLoader(ObjLoader const& other) = delete;
  ObjLoader(ObjLoader&& other)      = delete;

  ObjLoader& operator=(ObjLoader const& other) = delete;
  ObjLoader& operator=(ObjLoader&& other)      = delete;

  ~ObjLoader() = default;

  // Returns the stored vertices data of the .obj file, directly usable with gl draw arrays.
  std::shared_ptr<std::vector<float>> getVertices();

 private:
  void initData(const std::string& objFilePath);

  std::shared_ptr<std::vector<float>> mVertices;

};

} // namespace cs::graphics

#endif // CS_GRAPHICS_VISTAGLTF_HPP
