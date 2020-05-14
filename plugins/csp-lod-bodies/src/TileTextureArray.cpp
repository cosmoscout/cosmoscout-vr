////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TileTextureArray.hpp"

#include "RenderData.hpp"
#include "TreeManagerBase.hpp"

#include <VistaBase/VistaStreamUtils.h>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

// functions to obtain texture internal/external format and type
// from TileDataType value
GLenum getInternalFormat(TileDataType dataType) {
  GLenum result = GL_NONE;

  switch (dataType) {
  case TileDataType::eFloat32:
    result = GL_R32F;
    break;

  case TileDataType::eUInt8:
    result = GL_R8;
    break;

  case TileDataType::eU8Vec3:
    result = GL_RGB8;
    break;
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLenum getFormat(TileDataType dataType) {
  switch (dataType) {
  case TileDataType::eFloat32:
  case TileDataType::eUInt8:
    return GL_RED;
  case TileDataType::eU8Vec3:
    return GL_RGB;
  }

  return GL_NONE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLenum getType(TileDataType dataType) {
  switch (dataType) {
  case TileDataType::eFloat32:
    return GL_FLOAT;

  case TileDataType::eUInt8:
  case TileDataType::eU8Vec3:
    return GL_UNSIGNED_BYTE;
  }

  return GL_NONE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
TileTextureArray::TileTextureArray(TileDataType dataType, int maxLayerCount)
    : boost::noncopyable()
    , mTexId(0U)
    , mIformat()
    , mFormat()
    , mType()
    , mDataType(dataType)
    , mNumLayers(maxLayerCount) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileTextureArray::~TileTextureArray() {
  releaseTexture();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileTextureArray::allocateGPU(RenderData* rdata) {
  assert(rdata->getTexLayer() < 0);

  mUploadQueue.push_back(rdata);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileTextureArray::releaseGPU(RenderData* rdata) {
  if (rdata->getTexLayer() >= 0) {
    releaseLayer(rdata);
  } else {
    // Tile is not uploaded, could be in the queue?
    // XXX TODO Linear search, but mUploadQueue is usually small and
    //          this case should be rare

    auto rIt = std::find(mUploadQueue.begin(), mUploadQueue.end(), rdata);

    // avoid erasing an element in the middle of std::vector,
    // just invalidate the pointer and skip NULL entries when uploading
    if (rIt != mUploadQueue.end()) {
      *rIt = nullptr;
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

    RenderData* rdata = mUploadQueue.back();

    // rdata could be NULL if a tile is removed before it is ever
    // uploaded to the GPU, c.f. releaseGPU
    if (rdata != nullptr) {
      allocateLayer(rdata);
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
  GLsizei const width  = TileBase::SizeX;
  GLsizei const height = TileBase::SizeY;
  GLsizei const depth  = mNumLayers;
  GLint const   border = 0;

  mIformat = getInternalFormat(dataType);
  mFormat  = getFormat(dataType);
  mType    = getType(dataType);

  glBindTexture(GL_TEXTURE_2D_ARRAY, mTexId);
  glTexImage3D(
      GL_TEXTURE_2D_ARRAY, level, mIformat, width, height, depth, border, mFormat, mType, nullptr);

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

// Uploads tile data from the node associated with @a rdata to the GPU.
// @note May only be called after a call to @c preUpload.
void TileTextureArray::allocateLayer(RenderData* rdata) {
  assert(!mFreeLayers.empty());
  assert(rdata->getTexLayer() < 0);

  TileNode* node = rdata->getNode();
  TileBase* tile = node->getTile();

  int layer = mFreeLayers.back();
  mFreeLayers.pop_back();

  GLint const   level   = 0;
  GLint const   xoffset = 0;
  GLint const   yoffset = 0;
  GLsizei const width   = TileBase::SizeX;
  GLsizei const height  = TileBase::SizeY;
  GLsizei const depth   = 1;
  GLvoid const* data    = tile->getDataPtr();

  glTexSubImage3D(GL_TEXTURE_2D_ARRAY, level, xoffset, yoffset, layer, width, height, depth,
      mFormat, mType, data);

  rdata->setTexLayer(layer);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileTextureArray::releaseLayer(RenderData* rdata) {
  assert(rdata->getTexLayer() >= 0);

  // simply mark the layer as available and record that rdata is not
  // currently on the GPU (i.e. set the texture layer to an invalid value)
  mFreeLayers.push_back(rdata->getTexLayer());
  rdata->setTexLayer(-1);
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
