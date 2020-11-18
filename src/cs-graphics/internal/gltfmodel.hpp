////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_GLTFMODEL_HPP
#define CS_GRAPHICS_GLTFMODEL_HPP

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <tiny_gltf.h>
#include <vector>

namespace cs::graphics::internal {

struct GltfShared;

/// This contains all information about a single GLSL uniform variable.
struct UniformVar {
  std::string name;       ///< The name of the uniform variable.
  int         location{}; ///< The GLSL location of the uniform variable.
  int         size{};     ///< The size of the uniform. Uniforms that are not arrays will have
                          ///< the size of 1.
  unsigned int type{};    ///< The data type of the uniform.
};

/// This contains all information about a single GLSL texture variable.
struct TextureVar {
  std::string  name;       ///< The name of the texture variable.
  int          location{}; ///< The GLSL location of the texture variable.
  unsigned int unit{}; ///< The texture unit of the variable. E.g. GL_TEXTURE0, GL_TEXTURE1, etc.
  unsigned int type{}; ///< The data type of the texture. It will be one of the image types.
};

/// Contains all available infos about a GLSL program, including the uniforms and textures.
struct GLProgramInfo {
  /// Maps Attribute semantics such as "POSITION", "NORMAL", "TEXCOORD_0", to
  /// attribute locations e.g. 0 == pbr_attributes["POSITION"]
  std::map<std::string, int>        pbr_attributes;
  std::map<std::string, UniformVar> uniforms;
  std::map<std::string, TextureVar> textures;

  /// Executes a function with the UniformVar belonging to the given name.
  template <typename F>
  void withUniform(std::string const& name, F&& f) {
    auto it = uniforms.find(name);
    if (it != uniforms.end()) {
      f(it->second);
    }
  }

  int u_MVPMatrix_loc{};
  int u_ModelMatrix_loc{};
  int u_NormalMatrix_loc{};

  // Fragmentshader
  int u_LightDirection_loc{};
  int u_LightColor_loc{};
  int u_EnableHDR_loc{};

  int u_DiffuseEnvSampler_loc{};
  int u_SpecularEnvSampler_loc{};
  int u_brdfLUT_loc{};
  int u_IBLIntensity_loc{};
  int u_IBLrotation_loc{};

  int u_BaseColorSampler_loc{};
  int u_NormalSampler_loc{};
  int u_NormalScale_loc{};
  int u_EmissiveSampler_loc{};
  int u_EmissiveFactor_loc{};
  int u_MetallicRoughnessSampler_loc{};
  int u_OcclusionSampler_loc{};
  int u_OcclusionStrength_loc{};

  int u_MetallicRoughnessValues_loc{};
  int u_BaseColorFactor_loc{};

  int u_Camera_loc{};
  int u_FarClip_loc{};
};

/// Contains all available info about a texture.
struct Texture {
  unsigned int                  target;  ///< The texture target (E.g. GL_TEXTURE_2D).
  std::shared_ptr<unsigned int> sampler; ///< The pointer to a OpenGL sampler.
  std::shared_ptr<unsigned int> image;   ///< The pointer to a OpenGL texture.
};

/// Contains information about an OpenGL buffer object.
struct Buffer {
  unsigned int                  target{}; ///< The OpenGL target (E.g. GL_ARRAY_BUFFER).
  std::shared_ptr<unsigned int> id;       ///< The OpenGL buffer id.
};

/// An OpenGL primitive for rendering a vertex array.
struct Primitive {

  /// The rendering of the vertex array.
  void draw(glm::mat4 const& projMat, glm::mat4 const& viewMat, glm::mat4 const& modelMat,
      GltfShared const& shared) const;

  bool hasIndices = false;  ///< Determines if glDrawElements or glDrawArrays will be called.
  int  mode       = 0x0004; ///< GL_TRIANGLES;

  size_t verticesCount = 0;      ///< For glDrawArrays if (not hasIndices)
  size_t indicesCount  = 0;      ///< for glDrawElements if hasIndices
  int    indicesType   = 0x1403; ///< GL_UNSIGNED_SHORT;
  size_t byteOffset{};

  GLProgramInfo programInfo;

  glm::vec4 baseColorFactor{};
  glm::vec2 metallicRoughnessValues{}; ///< For physical based rendering.
  glm::vec3 emissiveFactor{};

  std::vector<std::pair<Texture, TextureVar>> textures;
  std::shared_ptr<unsigned int>               vaoPtr;
  std::shared_ptr<unsigned int>               programPtr;
};

/// Manages all primitives belonging to a mesh.
struct Mesh {
  void draw(glm::mat4 const& projMat, glm::mat4 const& viewMat, glm::mat4 const& modelMat,
      GltfShared const& shared) const;

  /// All primitives belonging to the model.
  std::vector<Primitive> primitives;

  /// minPos and maxPos define the bounding box of the mesh.
  glm::vec3 minPos = glm::vec3(std::numeric_limits<float>::lowest());
  glm::vec3 maxPos = glm::vec3(std::numeric_limits<float>::max());
};

/// Represents a GLTF model.
struct GltfShared {
  void init(tinygltf::Model const& gltf, const std::string& cubemapFilepath);

 private:
  void      buildMeshes(tinygltf::Model const& gltf);
  Primitive createMeshPrimitive(tinygltf::Model const& gltf, tinygltf::Primitive const& primitive);

 public:
  glm::vec3            m_lightColor     = glm::vec3(0.0F, 0.0F, 0.0F);
  glm::vec3            m_lightDirection = glm::vec3(0.0F, 0.0F, 1.0F);
  float                m_lightIntensity = 1.0F;
  bool                 m_enableHDR      = false;
  float                m_IBLIntensity   = 1.0F;
  glm::mat3            m_IBLrotation    = glm::mat3(1.0F);
  tinygltf::Model      mTinyGltfModel;
  std::vector<Texture> mTextures;
  std::vector<Mesh>    mMeshes;
  int                  mBrdfLUTindex        = -1;
  int                  mDiffuseEnvMapIndex  = -1;
  int                  mSpecularEnvMapIndex = -1;
  bool                 m_linearDepthBuffer  = false;
};

/// A Vista wrapper for the GLTF model responsible for rendering.
class VistaGltfNode : public IVistaOpenGLDraw {
 public:
  VistaGltfNode(tinygltf::Node const& node, std::shared_ptr<GltfShared> shared);
  ~VistaGltfNode() override;

  VistaGltfNode(VistaGltfNode const& other) = delete;
  VistaGltfNode(VistaGltfNode&& other)      = delete;

  VistaGltfNode& operator=(VistaGltfNode const& other) = delete;
  VistaGltfNode& operator=(VistaGltfNode&& other) = delete;

  /// The method Do() gets the callback from scene graph during the rendering process.
  bool Do() override;

  /// This method should return the bounding box of the OpenGL object you draw in the method Do().
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<GltfShared> mShared;
  std::string                 mName;
  int                         mMeshIndex;
  glm::vec3                   mMinPos = glm::vec3(std::numeric_limits<float>::lowest());
  glm::vec3                   mMaxPos = glm::vec3(std::numeric_limits<float>::max());
};

} // namespace cs::graphics::internal

#endif // CS_GRAPHICS_GLTFMODEL_HPP
