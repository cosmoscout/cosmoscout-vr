////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SinglePassRaycaster.hpp"

#include "../../../../../src/cs-core/Settings.hpp"
#include "../../../../../src/cs-core/SolarSystem.hpp"
#include "../../../../../src/cs-utils/utils.hpp"
#include "../../logger.hpp"

#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

SinglePassRaycaster::SinglePassRaycaster(std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::Settings>                                         settings)
    : mObjectName("None")
    , mSolarSystem(std::move(solarSystem))
    , mSettings(std::move(settings))
    , mTexture(GL_TEXTURE_3D)
    , mHasTexture(false)
    , mMinBounds(0)
    , mMaxBounds(0) {

  mTexture.Bind();
  mTexture.SetWrapS(GL_CLAMP_TO_EDGE);
  mTexture.SetWrapT(GL_CLAMP_TO_EDGE);
  mTexture.Unbind();

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::ePlanets) + 10);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SinglePassRaycaster::~SinglePassRaycaster() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SinglePassRaycaster::Do() {
  if (!mHasTexture || mObjectName == "None" || mObjectName.empty()) {
    return false;
  }

  auto object   = mSolarSystem->getObjectByCenterName(mObjectName);
  auto observer = mSolarSystem->getObserver();
  if (!object || object->getCenterName() != observer.getCenterName()) {
    return false;
  }




  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SinglePassRaycaster::GetBoundingBox(VistaBoundingBox& bb) {
  bb.SetBounds(glm::value_ptr(mMinBounds), glm::value_ptr(mMaxBounds));
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLenum get3DPixelFormat(size_t numComponents) {
  switch (numComponents) {
  case 1:
    return GL_RED;
  case 2:
    return GL_RG;
  case 3:
    return GL_RGB;
  default:
    return GL_RGBA;
  }
}

template <typename T>
std::vector<T> copyTo3DTextureBuffer(
    std::vector<std::vector<T>> const& imageData, size_t numScalars) {
  std::vector<T> data{};
  data.reserve(imageData.size() * std::min(numScalars, static_cast<size_t>(4)));

  for (auto const& point : imageData) {
    for (size_t i = 0; i < 4 && i < point.size(); i++) {
      data.emplace_back(point[i]);
    }
  }

  return data;
}

void SinglePassRaycaster::setData(std::shared_ptr<Volume3D> const& image) {
  mHasTexture = image != nullptr;

  if (!mHasTexture) {
    return;
  }

  mBounds            = image->mBounds;
  GLenum imageFormat = get3DPixelFormat(image->mNumScalars);

  mTexture.Bind();

  if (std::holds_alternative<U8ValueVector>(image->mPoints)) {
    auto imageData = std::get<U8ValueVector>(image->mPoints);

    std::vector<uint8_t> data = copyTo3DTextureBuffer(imageData, image->mNumScalars);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, image->mDimension.x, image->mDimension.y,
        image->mDimension.z, 0, imageFormat, GL_UNSIGNED_BYTE, data.data());

  } else if (std::holds_alternative<U16ValueVector>(image->mPoints)) {
    auto imageData = std::get<U16ValueVector>(image->mPoints);

    std::vector<uint16_t> data = copyTo3DTextureBuffer(imageData, image->mNumScalars);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, image->mDimension.x, image->mDimension.y,
        image->mDimension.z, 0, imageFormat, GL_UNSIGNED_SHORT, data.data());

  } else if (std::holds_alternative<U32ValueVector>(image->mPoints)) {
    auto imageData = std::get<U32ValueVector>(image->mPoints);

    std::vector<uint32_t> data = copyTo3DTextureBuffer(imageData, image->mNumScalars);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, image->mDimension.x, image->mDimension.y,
        image->mDimension.z, 0, imageFormat, GL_UNSIGNED_INT, data.data());

  } else if (std::holds_alternative<I16ValueVector>(image->mPoints)) {
    auto imageData = std::get<I16ValueVector>(image->mPoints);

    std::vector<int16_t> data = copyTo3DTextureBuffer(imageData, image->mNumScalars);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, image->mDimension.x, image->mDimension.y,
        image->mDimension.z, 0, imageFormat, GL_SHORT, data.data());

  } else if (std::holds_alternative<I32ValueVector>(image->mPoints)) {
    auto imageData = std::get<I32ValueVector>(image->mPoints);

    std::vector<int32_t> data = copyTo3DTextureBuffer(imageData, image->mNumScalars);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, image->mDimension.x, image->mDimension.y,
        image->mDimension.z, 0, imageFormat, GL_INT, data.data());

  } else if (std::holds_alternative<F32ValueVector>(image->mPoints)) {
    auto imageData = std::get<F32ValueVector>(image->mPoints);

    std::vector<float> data = copyTo3DTextureBuffer(imageData, image->mNumScalars);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_FLOAT, image->mDimension.x, image->mDimension.y,
        image->mDimension.z, 0, imageFormat, GL_INT, data.data());

  } else {
    logger().error("Unknown type!");
  }

  glTexParameteri(mTexture.GetTarget(), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  mTexture.Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SinglePassRaycaster::setCenter(std::string center) {
  mObjectName = std::move(center);
  if (mObjectName == "None" || mObjectName.empty()) {
    return;
  }

  auto object = mSolarSystem->getObjectByCenterName(mObjectName);
  if (!object) {
    return;
  }

  mMinBounds = -object->getRadii();
  mMaxBounds = object->getRadii();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string SinglePassRaycaster::getCenter() const {
  return mObjectName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery