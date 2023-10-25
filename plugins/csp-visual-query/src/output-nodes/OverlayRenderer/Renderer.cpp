////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Renderer.hpp"

#include "../../../../../src/cs-core/SolarSystem.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "VistaKernel/DisplayManager/VistaDisplayManager.h"
#include "VistaKernel/DisplayManager/VistaViewport.h"
#include "VistaKernel/GraphicsManager/VistaGroupNode.h"
#include "VistaKernel/GraphicsManager/VistaOpenGLNode.h"
#include "VistaKernel/GraphicsManager/VistaSceneGraph.h"
#include "VistaKernel/VistaSystem.h"
#include "VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h"

namespace csp::visualquery {

Renderer::Renderer(std::string objectName, std::shared_ptr<cs::core::SolarSystem> solarSystem)
    : mObjectName(std::move(objectName))
    , mSolarSystem(std::move(solarSystem))
    , mTexture(GL_TEXTURE_2D) {
  auto object = mSolarSystem->getObject(mObjectName);
  mMinBounds  = -object->getRadii();
  mMaxBounds  = object->getRadii();

  // create textures ---------------------------------------------------------
  for (auto const& viewport : GetVistaSystem()->GetDisplayManager()->GetViewports()) {
    // Texture for previous renderer depth buffer
    const auto [buffer, success] =
        mDepthBufferData.try_emplace(viewport.second, GL_TEXTURE_RECTANGLE);
    if (success) {
      buffer->second.Bind();
      buffer->second.SetWrapS(GL_CLAMP);
      buffer->second.SetWrapT(GL_CLAMP);
      buffer->second.SetMinFilter(GL_NEAREST);
      buffer->second.SetMagFilter(GL_NEAREST);
      buffer->second.Unbind();
    }
  }

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

Renderer::~Renderer() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

void Renderer::setData(Image2D image) {
  mBounds = image.mBounds;

  if (std::holds_alternative<U8ValueVector>(image.mPoints)) {
    auto imageData = std::get<U8ValueVector>(image.mPoints);

    std::vector<uint8_t> data{};
    data.reserve(imageData.size());

    for (auto const& point : imageData) {
      data.emplace_back(point.at(0));
      data.emplace_back(point.at(0));
      data.emplace_back(point.at(0));
      data.emplace_back(255U);
    }

    mTexture.UploadTexture(image.mDimension.x, image.mDimension.y, data.data(), false, GL_RGBA, GL_UNSIGNED_BYTE);
  } else if (std::holds_alternative<U16ValueVector>(image.mPoints)) {
    auto imageData = std::get<U16ValueVector>(image.mPoints);
    // TODO
  } else if (std::holds_alternative<U32ValueVector>(image.mPoints)) {
    auto imageData = std::get<U32ValueVector>(image.mPoints);
    // TODO
  } else if (std::holds_alternative<I16ValueVector>(image.mPoints)) {
    auto imageData = std::get<I16ValueVector>(image.mPoints);
    // TODO
  } else if (std::holds_alternative<I32ValueVector>(image.mPoints)) {
    auto imageData = std::get<I32ValueVector>(image.mPoints);
    // TODO
  } else if (std::holds_alternative<F32ValueVector>(image.mPoints)) {
    auto imageData = std::get<F32ValueVector>(image.mPoints);

    std::vector<float> data{};
    data.reserve(imageData.size());

    for (auto const& point : imageData) {
      data.emplace_back(point.at(0));
      data.emplace_back(point.at(0));
      data.emplace_back(point.at(0));
      data.emplace_back(1.0F);
    }

    mTexture.UploadTexture(image.mDimension.x, image.mDimension.y, data.data(), false, GL_RGBA, GL_FLOAT);
  } else {
    logger().error("Unknown type!");
  }

}

bool Renderer::Do() {
  if (mShaderDirty) {
    mShader = VistaGLSLShader();

    mShader.InitGeometryShaderFromString(SURFACE_GEOM);
    mShader.InitVertexShaderFromString(SURFACE_VERT);
    mShader.InitFragmentShaderFromString(SURFACE_FRAG);
    mShader.Link();

    mShaderDirty = false;
  }


  return true;
}

bool Renderer::GetBoundingBox(VistaBoundingBox& bb) {
  bb.SetBounds(glm::value_ptr(mMinBounds), glm::value_ptr(mMaxBounds));
  return true;
}

} // namespace csp::visualquery