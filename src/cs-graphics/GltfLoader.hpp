////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_VISTAGLTF_HPP
#define CS_GRAPHICS_VISTAGLTF_HPP

#include "cs_graphics_export.hpp"

#include <VistaBase/VistaColor.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

#include <glm/glm.hpp>
#include <map>
#include <memory>
#include <vector>

class VistaVector3D;
class VistaTransformNode;
class VistaSceneGraph;

namespace tinygltf {
struct Model;
struct Node;
} // namespace tinygltf

namespace cs::graphics {

namespace internal {
struct GltfShared;
}

/// If added to the scene graph, this will draw a Gltf 2.0 model.
// TODO maybe rename to GltfModel, because it does a lot more than loading an gltf model.
class CS_GRAPHICS_EXPORT GltfLoader {
 public:
  /// Creates a gltf model from the gltf and cubemap files.
  GltfLoader(const std::string& sGltfFile, const std::string& cubemapFilepath,
      bool linearDepthBuffer = false);

  GltfLoader(GltfLoader const& other) = delete;
  GltfLoader(GltfLoader&& other)      = delete;

  GltfLoader& operator=(GltfLoader const& other) = delete;
  GltfLoader& operator=(GltfLoader&& other) = delete;

  ~GltfLoader() = default;

  void setLightColor(float r, float g, float b);
  void setLightDirection(float x, float y, float z);
  void setLightDirection(VistaVector3D const& dir);
  void setLightIntensity(float intensity);
  void setEnableHDR(bool enable);

  void setIBLIntensity(float intensity);

  /// Please make sure that the supplied matrix is orthogonal.
  ///   transpose(m) == inverse(m);
  void rotateIBL(glm::mat3 const& m);

  /// Attaches the model to the VistaSceneGraph for rendering.
  bool attachTo(VistaSceneGraph* sg, VistaTransformNode* parent);

 private:
  std::shared_ptr<internal::GltfShared> mShared;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_VISTAGLTF_HPP
