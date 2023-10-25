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
  std::vector<float> data{};
  data.reserve(image.getPoints().size());

  mBounds = csl::ogc::Bounds(image.getBound("lon")->min, image.getBound("lon")->max,
      image.getBound("lat")->min, image.getBound("lat")->max);

  for (auto const& point : image.getPoints()) {
    data.emplace_back(point.value.at(0));
    data.emplace_back(point.value.at(0));
    data.emplace_back(point.value.at(0));
  }

  mTexture.UploadTexture(image.mDimension.getWidth(), image.mDimension.getLength(), data.data(), false);
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