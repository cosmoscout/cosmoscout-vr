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

Renderer::Renderer(std::string objectName, std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::Settings> settings)
    : mObjectName(std::move(objectName))
    , mSolarSystem(std::move(solarSystem))
    , mSettings(std::move(settings))
    , mTexture(GL_TEXTURE_2D)
    , mHasTexture(false) {
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

GLenum getPixelFormat(size_t numComponents) {
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
std::vector<T> copyToTextureBuffer(
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

void Renderer::setData(std::shared_ptr<Image2D> const& image) {
  mHasTexture = image != nullptr;

  if (!mHasTexture) {
    return;
  }

  mBounds            = image->mBounds;
  GLenum imageFormat = getPixelFormat(image->mNumScalars);

  if (std::holds_alternative<U8ValueVector>(image->mPoints)) {
    auto                 imageData = std::get<U8ValueVector>(image->mPoints);
    std::vector<uint8_t> data      = copyToTextureBuffer(imageData, image->mNumScalars);
    mTexture.UploadTexture(image->mDimension.x, image->mDimension.y, data.data(), false,
        imageFormat, GL_UNSIGNED_BYTE);

  } else if (std::holds_alternative<U16ValueVector>(image->mPoints)) {
    auto                  imageData = std::get<U16ValueVector>(image->mPoints);
    std::vector<uint16_t> data      = copyToTextureBuffer(imageData, image->mNumScalars);
    mTexture.UploadTexture(image->mDimension.x, image->mDimension.y, data.data(), false,
        imageFormat, GL_UNSIGNED_SHORT);

  } else if (std::holds_alternative<U32ValueVector>(image->mPoints)) {
    auto                  imageData = std::get<U32ValueVector>(image->mPoints);
    std::vector<uint32_t> data      = copyToTextureBuffer(imageData, image->mNumScalars);
    mTexture.UploadTexture(
        image->mDimension.x, image->mDimension.y, data.data(), false, imageFormat, GL_UNSIGNED_INT);

  } else if (std::holds_alternative<I16ValueVector>(image->mPoints)) {
    auto                 imageData = std::get<I16ValueVector>(image->mPoints);
    std::vector<int16_t> data      = copyToTextureBuffer(imageData, image->mNumScalars);
    mTexture.UploadTexture(
        image->mDimension.x, image->mDimension.y, data.data(), false, imageFormat, GL_SHORT);

  } else if (std::holds_alternative<I32ValueVector>(image->mPoints)) {
    auto                 imageData = std::get<I32ValueVector>(image->mPoints);
    std::vector<int32_t> data      = copyToTextureBuffer(imageData, image->mNumScalars);
    mTexture.UploadTexture(
        image->mDimension.x, image->mDimension.y, data.data(), false, imageFormat, GL_INT);

  } else if (std::holds_alternative<F32ValueVector>(image->mPoints)) {
    auto               imageData = std::get<F32ValueVector>(image->mPoints);
    std::vector<float> data      = copyToTextureBuffer(imageData, image->mNumScalars);
    mTexture.UploadTexture(
        image->mDimension.x, image->mDimension.y, data.data(), false, imageFormat, GL_FLOAT);

  } else {
    logger().error("Unknown type!");
  }
}

bool Renderer::Do() {
  if (!mHasTexture) {
    return false;
  }

  if (mShaderDirty) {
    mShader = VistaGLSLShader();

    mShader.InitGeometryShaderFromString(SURFACE_GEOM);
    mShader.InitVertexShaderFromString(SURFACE_VERT);
    mShader.InitFragmentShaderFromString(SURFACE_FRAG);
    mShader.Link();

    mShaderDirty = false;
  }

  glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT);
  glEnable(GL_TEXTURE_2D);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // copy depth buffer from previous rendering
  // -------------------------------------------------------
  std::array<GLint, 4> iViewport{};
  glGetIntegerv(GL_VIEWPORT, iViewport.data());

  auto* viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  VistaTexture& depthBuffer = mDepthBufferData.at(viewport);

  depthBuffer.Bind();
  glCopyTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT, iViewport[0], iViewport[1],
      iViewport[2], iViewport[3], 0);

  auto object    = mSolarSystem->getObject(mObjectName);
  auto radii     = object->getRadii();
  auto transform = object->getObserverRelativeTransform();

  // get matrices and related values
  std::array<GLfloat, 16> glMatV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matV = glm::make_mat4x4(glMatV.data());
  auto matP = glm::make_mat4x4(glMatP.data());

  glm::dmat4 matInvMVP = glm::inverse(glm::dmat4(matP) * glm::dmat4(matV) * transform);

  // Bind shader before draw
  mShader.Bind();

  // Only bind the enabled textures.
  depthBuffer.Bind(GL_TEXTURE0);
  mTexture.Bind(GL_TEXTURE1);

  mShader.SetUniform(mShader.GetUniformLocation("uDepthBuffer"), 0);
  mShader.SetUniform(mShader.GetUniformLocation("uTexture"), 1);

  GLint loc = mShader.GetUniformLocation("uMatInvMVP");
  glUniformMatrix4dv(loc, 1, GL_FALSE, glm::value_ptr(matInvMVP));

  // Double precision bounds
  loc = mShader.GetUniformLocation("uLatRange");
  glUniform2dv(loc, 1,
      glm::value_ptr(cs::utils::convert::toRadians(glm::dvec2(mBounds.mMinLat, mBounds.mMaxLat))));
  loc = mShader.GetUniformLocation("uLonRange");
  glUniform2dv(loc, 1,
      glm::value_ptr(cs::utils::convert::toRadians(glm::dvec2(mBounds.mMinLon, mBounds.mMaxLon))));

  glm::vec3 sunDirection(1, 0, 0);
  float     sunIlluminance(1.F);
  float     ambientBrightness(mSettings->mGraphics.pAmbientBrightness.get());

  if (object == mSolarSystem->getSun()) {
    // If the overlay is on the sun, we have to calculate the lighting differently.
    if (mSettings->mGraphics.pEnableHDR.get()) {
      // The variable is called illuminance, for the sun it contains actually luminance values.
      sunIlluminance = static_cast<float>(mSolarSystem->getSunLuminance());

      // For planets, this illuminance is divided by pi, so we have to premultiply it for the sun.
      sunIlluminance *= glm::pi<float>();
    }

    ambientBrightness = 1.0F;
  } else {
    // For all other bodies we can use the utility methods from the SolarSystem.
    if (mSettings->mGraphics.pEnableHDR.get()) {
      sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(transform[3]));
    }

    sunDirection = glm::normalize(
        glm::inverse(transform) * glm::dvec4(mSolarSystem->getSunDirection(transform[3]), 0.0));
  }

  mShader.SetUniform(mShader.GetUniformLocation("uSunDirection"), sunDirection[0], sunDirection[1],
      sunDirection[2]);
  mShader.SetUniform(mShader.GetUniformLocation("uSunIlluminance"), sunIlluminance);
  mShader.SetUniform(mShader.GetUniformLocation("uAmbientBrightness"), ambientBrightness);

  // provide radii to shader
  mShader.SetUniform(mShader.GetUniformLocation("uRadii"), static_cast<float>(radii[0]),
      static_cast<float>(radii[1]), static_cast<float>(radii[2]));

  int depthBits = 0;
  glGetIntegerv(GL_DEPTH_BITS, &depthBits);

  // Dummy draw
  glDrawArrays(GL_POINTS, 0, 1);

  depthBuffer.Unbind(GL_TEXTURE0);
  mTexture.Unbind(GL_TEXTURE1);

  mShader.Release();

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glPopAttrib();

  return true;
}

bool Renderer::GetBoundingBox(VistaBoundingBox& bb) {
  bb.SetBounds(glm::value_ptr(mMinBounds), glm::value_ptr(mMaxBounds));
  return true;
}

} // namespace csp::visualquery