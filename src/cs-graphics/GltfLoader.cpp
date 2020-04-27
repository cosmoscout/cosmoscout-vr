////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GltfLoader.hpp"

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>
#endif

#include <VistaBase/VistaVector3D.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/GraphicsManager/VistaGeometryFactory.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>

#include <VistaInterProcComm/Connections/VistaByteBufferDeSerializer.h>

#include <fstream>
#include <glm/gtc/matrix_transform.hpp>

#include "internal/gltfmodel.hpp"

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool FromString(std::string const& v, T& out) {
  std::istringstream iss(v);
  iss >> out;
  return (iss.rdstate() & std::stringstream::failbit) == 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string GetFilePathExtension(const std::string& FileName) {
  if (FileName.find_last_of('.') != std::string::npos) {
    return FileName.substr(FileName.find_last_of('.') + 1);
  }

  return "";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

GltfLoader::GltfLoader(
    const std::string& sGltfFile, const std::string& cubemapFilepath, bool linearDepthBuffer)
    : mShared(std::make_shared<internal::GltfShared>()) {
  tinygltf::TinyGLTF loader;
  std::string        err;
  std::string        warn;
  std::string        ext = GetFilePathExtension(sGltfFile);

  bool ret = false;
  if (ext == "glb") {
    // Assume binary glTF.
    ret = loader.LoadBinaryFromFile(&mShared->mTinyGltfModel, &err, &warn, sGltfFile);
  } else {
    // Assume ascii glTF.
    ret = loader.LoadASCIIFromFile(&mShared->mTinyGltfModel, &err, &warn, sGltfFile);
  }

  if (!err.empty()) {
    throw std::runtime_error(err);
  }
  if (!ret) {
    std::string msg("Failed to load .glTF: ");
    throw std::runtime_error(msg + sGltfFile);
  }

  mShared->m_linearDepthBuffer = linearDepthBuffer;
  mShared->init(mShared->mTinyGltfModel, cubemapFilepath);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GltfLoader::rotateIBL(glm::mat3 const& m) {
  mShared->m_IBLrotation = m;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GltfLoader::setLightColor(float r, float g, float b) {
  mShared->m_lightColor = glm::vec3(r, g, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GltfLoader::setLightDirection(float x, float y, float z) {
  mShared->m_lightDirection = glm::vec3(x, y, z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GltfLoader::setLightDirection(VistaVector3D const& dir) {
  setLightDirection(dir[0], dir[1], dir[2]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GltfLoader::setLightIntensity(float intensity) {
  mShared->m_lightIntensity = intensity;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GltfLoader::setEnableHDR(bool enable) {
  mShared->m_enableHDR = enable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GltfLoader::setIBLIntensity(float intensity) {
  mShared->m_IBLIntensity = intensity;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void apply_transform(VistaTransformNode& vista_transform, tinygltf::Node const& node) {
  if (node.matrix.size() == 16) {
    glm::mat4 mat = glm::make_mat4(node.matrix.data());
    vista_transform.SetTransform(VistaTransformMatrix(mat[0][0], mat[1][0], mat[2][0], mat[3][0],
        mat[0][1], mat[1][1], mat[2][1], mat[3][1], mat[0][2], mat[1][2], mat[2][2], mat[3][2],
        mat[0][3], mat[1][3], mat[2][3], mat[3][3]));
  } else {
    // Assume Trans x Rotate x Scale order
    if (node.scale.size() == 3) {
      vista_transform.SetScale(static_cast<float>(node.scale[0]), static_cast<float>(node.scale[1]),
          static_cast<float>(node.scale[1]));
    }

    if (node.rotation.size() == 4) {
      vista_transform.Rotate(node.rotation.data());
    }

    if (node.translation.size() == 3) {
      vista_transform.Translate(node.translation.data());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void build_node(VistaSceneGraph& sg, std::shared_ptr<internal::GltfShared> const& shared,
    VistaTransformNode* parent, tinygltf::Node const& tinygltf_node) {
  VistaTransformNode* transform_node = sg.NewTransformNode(parent);
  if (tinygltf_node.mesh >= 0) {
    auto* draw = new internal::VistaGltfNode(tinygltf_node, shared);
    sg.NewOpenGLNode(transform_node, draw);
  }

  apply_transform(*transform_node, tinygltf_node);
  transform_node->SetName(tinygltf_node.name);
  for (int i : tinygltf_node.children) {
    build_node(sg, shared, transform_node, shared->mTinyGltfModel.nodes[i]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool GltfLoader::attachTo(VistaSceneGraph* pSG, VistaTransformNode* parent) {
  if (mShared->mTinyGltfModel.scenes.empty()) {
    return false;
  }

  auto const& scene = (mShared->mTinyGltfModel.defaultScene >= 0)
                          ? mShared->mTinyGltfModel.scenes[mShared->mTinyGltfModel.defaultScene]
                          : mShared->mTinyGltfModel.scenes.front();

  for (int i : scene.nodes) {
    build_node(*pSG, mShared, parent, mShared->mTinyGltfModel.nodes[i]);
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
