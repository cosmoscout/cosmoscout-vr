////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TILETEXTUREARRAY_HPP
#define CSP_LOD_BODIES_TILETEXTUREARRAY_HPP

#include "TileDataType.hpp"

#include <GL/glew.h>
#include <array>
#include <boost/noncopyable.hpp>
#include <memory>
#include <vector>

namespace csp::lodbodies {

class TileNode;
class RenderData;
class TreeManagerBase;

/// Responsible for handling tile data that is uploaded the GPU and for balancing additional upload
/// requests.
///
/// Tile data is stored in a 2D array texture (GL_TEXTURE_2D_ARRAY) with width and height matching
/// those of a single tile and as many layers as there are tiles that can be kept on the GPU at
/// once. If more tiles are needed on the GPU the array texture must be resized (which requires all
/// tiles to be re-uploaded), this should be avoided to prevent the tile resolution to drop
/// dramatically while only low resolution tiles are on the GPU.
class TileTextureArray : private boost::noncopyable {
 public:
  explicit TileTextureArray(TileDataType dataType, int maxLayerCount);

  TileTextureArray(TileTextureArray const& other) = delete;
  TileTextureArray(TileTextureArray&& other)      = delete;

  TileTextureArray& operator=(TileTextureArray const& other) = delete;
  TileTextureArray& operator=(TileTextureArray&& other) = delete;

  ~TileTextureArray();

  /// Requests that data for the tile associated with rdata be uploaded to the GPU.
  void allocateGPU(RenderData* rdata);

  /// Release GPU resources allocated for the tile associated with rdata.
  void releaseGPU(RenderData* rdata);

  /// Process up to maxItems upload requests.
  void processQueue(int maxItems);

  /// Returns the OpenGL id of the texture used to store tiles on the GPU. This is an internal
  /// interface for TileRenderer.
  unsigned int getTextureId() const;

  /// Gets Total Layer Count
  std::size_t getTotalLayerCount() const;

  /// Gets Used Layer Count
  std::size_t getUsedLayerCount() const;

 private:
  void allocateTexture(TileDataType dataType);
  void releaseTexture();

  void allocateLayer(RenderData* rdata);
  void releaseLayer(RenderData* rdata);

  void        preUpload();
  static void postUpload();

  GLuint       mTexId;
  GLenum       mIformat;
  GLenum       mFormat;
  GLenum       mType;
  TileDataType mDataType;

  const GLint        mNumLayers;
  std::vector<GLint> mFreeLayers;

  std::vector<RenderData*> mUploadQueue;
};

/// DocTODO
class GLResources {
 public:
  GLResources(int maxLayersFloat32, int maxLayersUInt8, int maxLayersU8Vec3) {
    mextureArrays[static_cast<int>(TileDataType::eFloat32)] =
        std::make_unique<TileTextureArray>(TileDataType::eFloat32, maxLayersFloat32);
    mextureArrays[static_cast<int>(TileDataType::eUInt8)] =
        std::make_unique<TileTextureArray>(TileDataType::eUInt8, maxLayersUInt8);
    mextureArrays[static_cast<int>(TileDataType::eU8Vec3)] =
        std::make_unique<TileTextureArray>(TileDataType::eU8Vec3, maxLayersU8Vec3);
  }

  TileTextureArray& operator[](TileDataType type) {
    return *mextureArrays.at(static_cast<int>(type));
  }

 private:
  std::array<std::unique_ptr<TileTextureArray>, 3> mextureArrays;
};
} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILETEXTUREARRAY_HPP
