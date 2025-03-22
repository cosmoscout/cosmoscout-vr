////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TileTextureArray.hpp"

#include "BaseTileData.hpp"
#include "TreeManager.hpp"

#include <VistaBase/VistaStreamUtils.h>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

// functions to obtain texture internal/external format and type
// from TileDataType value
GLenum getInternalFormat(TileDataType dataType) {
  GLenum result = GL_NONE;

  switch (dataType) {
  case TileDataType::eElevation:
    result = GL_R32F;
    break;

  case TileDataType::eColor:
    result = GL_RGBA8;
    break;
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLenum getFormat(TileDataType dataType) {
  switch (dataType) {
  case TileDataType::eElevation:
    return GL_RED;
  case TileDataType::eColor:
    return GL_RGBA;
  }

  return GL_NONE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLenum getType(TileDataType dataType) {
  switch (dataType) {
  case TileDataType::eElevation:
    return GL_FLOAT;

  case TileDataType::eColor:
    return GL_UNSIGNED_BYTE;
  }

  return GL_NONE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
TileTextureArray::TileTextureArray(TileDataType dataType, int maxLayerCount, uint32_t resolution)
    : mTexId(0U)
    , mIformat()
    , mFormat()
    , mType()
    , mDataType(dataType)
    , mResolution(resolution)
    , mNumLayers(maxLayerCount) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileTextureArray::~TileTextureArray() {
  releaseTexture();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileDataType TileTextureArray::getDataType() const {
  return mDataType;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileTextureArray::allocateGPU(std::shared_ptr<BaseTileData> data) {
  mUploadQueue.push_back(std::move(data));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileTextureArray::releaseGPU(std::shared_ptr<BaseTileData> const& data) {

  if (data->getTexLayer() >= 0) {
    releaseLayer(data);
  } else {
    // TileData is not uploaded, could be in the queue?
    // XXX TODO Linear search, but mUploadQueue is usually small and
    //          this case should be rare
    auto rIt = std::find(mUploadQueue.begin(), mUploadQueue.end(), data);

    // avoid erasing an element in the middle of std::vector,
    // just invalidate the pointer and skip NULL entries when uploading
    if (rIt != mUploadQueue.end()) {
      rIt->reset();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileTextureArray::processQueue(int maxItems) {
  if (mUploadQueue.empty()) {
    return;
  }

  preUpload();

  std::size_t upload = std::min<std::size_t>(maxItems, mUploadQueue.size());

  if (mFreeLayers.size() < upload) {
    // XXX TODO This is bad for performance and visuals, since *all* tiles
    // must be (re)-uploaded to the larger texture
    vstr::warnp() << "[TileTextureArray::processQueue]"
                  << " Pre-allocated GPU storage exhausted!"
                  << " [" << getUsedLayerCount() << " | " << getTotalLayerCount() << "]"
                  << std::endl;
  }

  int count = 0;

  while (!mUploadQueue.empty() && count < maxItems) {
    if (mFreeLayers.empty()) {
      break;
    }

    auto data = mUploadQueue.back();

    // data could be NULL if a tile is removed before it is ever
    // uploaded to the GPU, c.f. releaseGPU
    if (data) {
      allocateLayer(data);
      ++count;
    }

    mUploadQueue.pop_back();
  }

  postUpload();

  if (count > 0) {
#if !defined(NDEBUG) && !defined(VISTAPLANET_NO_VERBOSE)
    vstr::outi() << "[TileTextureArray::processQueue]"
                 << " uploaded/pending/used/free layers " << count << " / " << mUploadQueue.size()
                 << " / " << (mNumLayers - mFreeLayers.size()) << " / " << mFreeLayers.size()
                 << std::endl;
#endif
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

unsigned int TileTextureArray::getTextureId() const {
  return mTexId;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::size_t TileTextureArray::getTotalLayerCount() const {
  return mNumLayers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::size_t TileTextureArray::getUsedLayerCount() const {
  return mNumLayers - mFreeLayers.size();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileTextureArray::allocateTexture(TileDataType dataType) {
  if (mTexId > 0U) {
    return;
  }

  // allocate a 2D array texture for storing tile data of type dataType

  glGenTextures(1, &mTexId);

  GLsizei const level  = 0;
  GLsizei const depth  = mNumLayers;
  GLint const   border = 0;

  mIformat = getInternalFormat(dataType);
  mFormat  = getFormat(dataType);
  mType    = getType(dataType);

  glBindTexture(GL_TEXTURE_2D_ARRAY, mTexId);
  glTexImage3D(GL_TEXTURE_2D_ARRAY, level, mIformat, mResolution, mResolution, depth, border,
      mFormat, mType, nullptr);

  // set filter and wrapping parameters
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glBindTexture(GL_TEXTURE_2D_ARRAY, 0U);

  // all layers of newly allocated texture are available for use
  mFreeLayers.reserve(mNumLayers);

  for (int i = 0; i < mNumLayers; ++i) {
    mFreeLayers.push_back(mNumLayers - i - 1);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileTextureArray::releaseTexture() {
  if (mTexId == 0U) {
    return;
  }

  glDeleteTextures(1, &mTexId);
  mTexId = 0U;
  mFreeLayers.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Uploads tile data from the node associated with @a data to the GPU.
// @note May only be called after a call to @c preUpload.
void TileTextureArray::allocateLayer(std::shared_ptr<BaseTileData> const& data) {
  assert(!mFreeLayers.empty());
  assert(data->getTexLayer() < 0);

  int layer = mFreeLayers.back();
  mFreeLayers.pop_back();

  GLint const   level   = 0;
  GLint const   xoffset = 0;
  GLint const   yoffset = 0;
  GLsizei const depth   = 1;
  GLvoid const* pixels  = data->getDataPtr();

  glTexSubImage3D(GL_TEXTURE_2D_ARRAY, level, xoffset, yoffset, layer, mResolution, mResolution,
      depth, mFormat, mType, pixels);

  data->setTexLayer(layer);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileTextureArray::releaseLayer(std::shared_ptr<BaseTileData> const& data) {
  assert(data->getTexLayer() >= 0);

  // simply mark the layer as available and record that data is not
  // currently on the GPU (i.e. set the texture layer to an invalid value)
  mFreeLayers.push_back(data->getTexLayer());
  data->setTexLayer(-1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileTextureArray::preUpload() {
  allocateTexture(mDataType);
  glBindTexture(GL_TEXTURE_2D_ARRAY, mTexId);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileTextureArray::postUpload() {
  glBindTexture(GL_TEXTURE_2D_ARRAY, 0U);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
