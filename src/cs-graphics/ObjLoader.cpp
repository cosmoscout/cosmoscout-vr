////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ObjLoader.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

ObjLoader::ObjLoader(const std::string& objFilePath) {
  initData(objFilePath);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<std::vector<float>> ObjLoader::getVertices() {
  return mVertices;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ObjLoader::initData(const std::string& objFilePath) {

  // Open the .obj file.
  std::ifstream file(objFilePath);
  if (!file.is_open()) {
    std::string msg("Failed to load .obj: ");
    throw std::runtime_error(msg + objFilePath);
    return;
  }

  std::vector<float> vertices;
  struct TempVertex { float x, y, z; };
  std::vector<TempVertex> tempVertices;
  std::string line;

  // Read every line in the file.
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string lineHeader;
    ss >> lineHeader;

    // Reads the vertex lines and remember the vertices in the line in the TempVertex struct.
    if (lineHeader == "v") {
      TempVertex v;
      ss >> v.x >> v.y >> v.z;
      tempVertices.push_back(v);

    // Read the faces lines representing the vertex indices and writes the vertexes at these
    // indices in the final vertex array returned in correct order.
    } else if (lineHeader =="f") {
      std::string vertexStr;
    
      while (ss >> vertexStr) {
        int index = std::stoi(vertexStr) - 1;

        vertices.push_back(tempVertices[index].x);
        vertices.push_back(tempVertices[index].y);
        vertices.push_back(tempVertices[index].z);
      }
    }
  }
    
  // Close the read file again.
  file.close();
  
  // Set the read vertices in this object.
  mVertices = std::make_shared<std::vector<float>>(vertices);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
