////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILETEXTUREARRAY_HPP
#define CSP_LOD_BODIES_TILETEXTUREARRAY_HPP

#include "TileDataType.hpp"

#include <GL/glew.h>
#include <array>
#include <memory>
#include <vector>

namespace csp::lodbodies {

class TileNode;
class TileDataBase;
class TreeManager;

/// Responsible for handling tile data that is uploaded the GPU and for balancing additional upload
/// requests.
///
/// Tile data is stored in a 2D array texture (GL_TEXTURE_2D_ARRAY) with width and height matching
/// those of a single tile and as many layers as there are tiles that can be kept on the GPU at
/// once. If more tiles are needed on the GPU the array texture must be resized (which requires all
/// tiles to be re-uploaded), this should be avoided to prevent the tile resolution to drop
/// dramatically while only low resolution tiles are on the GPU.
class TileTextureArray {
 public:
  explicit TileTextureArray(TileDataType dataType, int maxLayerCount, uint32_t resolution);

  TileTextureArray(TileTextureArray const& other) = delete;
  TileTextureArray(TileTextureArray&& other)      = delete;

  TileTextureArray& operator=(TileTextureArray const& other) = delete;
  TileTextureArray& operator=(TileTextureArray&& other) = delete;

  ~TileTextureArray();

  TileDataType getDataType() const;

  /// Requests that data for the tile associated with data be uploaded to the GPU.
  void allocateGPU(std::shared_ptr<TileDataBase> data);

  /// Release GPU resources allocated for the tile associated with data.
  void releaseGPU(std::shared_ptr<TileDataBase> const& data);

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

  void allocateLayer(std::shared_ptr<TileDataBase> const& data);
  void releaseLayer(std::shared_ptr<TileDataBase> const& data);

  void        preUpload();
  static void postUpload();

  GLuint       mTexId;
  GLenum       mIformat;
  GLenum       mFormat;
  GLenum       mType;
  TileDataType mDataType;
  uint32_t     mResolution;

  const GLint        mNumLayers;
  std::vector<GLint> mFreeLayers;

  std::vector<std::shared_ptr<TileDataBase>> mUploadQueue;
};

/// DocTODO
class GLResources : public PerDataType<std::unique_ptr<TileTextureArray>> {
 public:
  GLResources(int maxElevationLayers, int maxColorLayers, uint32_t elevationResolution,
      uint32_t colorResolution)
      : PerDataType<std::unique_ptr<TileTextureArray>>(
            {std::make_unique<TileTextureArray>(
                 TileDataType::eElevation, maxElevationLayers, elevationResolution),
                std::make_unique<TileTextureArray>(
                    TileDataType::eColor, maxColorLayers, colorResolution)}) {
  }
};
} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILETEXTUREARRAY_HPP
