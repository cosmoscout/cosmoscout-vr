////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Renderer.hpp"

#include "../../../../../src/cs-core/Settings.hpp"
#include "../../../../../src/cs-core/SolarSystem.hpp"
#include "../../../../../src/cs-utils/utils.hpp"
#include "../../logger.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::visualquery {

namespace {
GLenum getInputFormat(size_t numComponents) {
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

GLenum getDataType(int dataType) {
  switch (dataType) {
  case 1:
    return GL_UNSIGNED_BYTE;
  case 2:
    return GL_UNSIGNED_SHORT;
  case 3:
    return GL_SHORT;
  case 4:
    return GL_UNSIGNED_INT;
  case 5:
    return GL_INT;
  default:
    return GL_FLOAT;
  }
}

GLenum getInternalFormat(size_t numComponents, int dataType) {
  // Allowed data types:
  //  GDT_Byte = 1
  //  GDT_UInt16 = 2
  //  GDT_Int16 = 3
  //  GDT_UInt32 = 4
  //  GDT_Int32 = 5
  //  GDT_Float32 = 6

  if (numComponents > 4) {
    logger().error("Unsupported number of components!");
    return GL_RGBA32F;
  }

  if (dataType < 1 || dataType > 6) {
    logger().error("Unsupported data type!");
    return GL_RGBA32F;
  }

  std::array<std::array<GLenum, 6>, 4> internalFormats = {{
      {GL_R8, GL_R16UI, GL_R16, GL_R32UI, GL_R32I, GL_R32F},
      {GL_RG8, GL_RG16UI, GL_RG16, GL_RG32UI, GL_RG32I, GL_RG32F},
      {GL_RGB8, GL_RGB16UI, GL_RGB16, GL_RGB32UI, GL_RGB32I, GL_RGB32F},
      {GL_RGBA8, GL_RGBA16UI, GL_RGBA16, GL_RGBA32UI, GL_RGBA32I, GL_RGBA32F},
  }};

  return internalFormats[numComponents - 1][dataType - 1];
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

template <typename T>
void uploadTextureData(VistaTexture& texture, std::vector<std::vector<T>> const& imageData,
    std::shared_ptr<Image2D> const& image) {
  std::vector<T> data = copyToTextureBuffer(imageData, image->mNumScalars);

  int dataType = 0;

  if (std::holds_alternative<U8ValueVector>(image->mPoints)) {
    dataType = 1;
  } else if (std::holds_alternative<U16ValueVector>(image->mPoints)) {
    dataType = 2;
  } else if (std::holds_alternative<I16ValueVector>(image->mPoints)) {
    dataType = 4;
  } else if (std::holds_alternative<U32ValueVector>(image->mPoints)) {
    dataType = 3;
  } else if (std::holds_alternative<I32ValueVector>(image->mPoints)) {
    dataType = 5;
  } else if (std::holds_alternative<F32ValueVector>(image->mPoints)) {
    dataType = 6;
  } else {
    logger().error("Unknown type!");
  }

  GLenum inputFormat    = getInputFormat(image->mNumScalars);
  GLenum inputType      = getDataType(dataType);
  GLenum internalFormat = getInternalFormat(image->mNumScalars, dataType);

  texture.Bind();
  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, image->mDimension.x, image->mDimension.y, 0,
      inputFormat, inputType, data.data());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  texture.Unbind();
}

} // namespace

Renderer::Renderer(std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::Settings>                   settings)
    : mObjectName("None")
    , mSolarSystem(std::move(solarSystem))
    , mSettings(std::move(settings))
    , mTexture(GL_TEXTURE_2D)
    , mHasTexture(false)
    , mLUT(GL_TEXTURE_1D)
    , mMinBounds(0)
    , mMaxBounds(0) {

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

  mLUT.Bind();
  mLUT.SetWrapS(GL_CLAMP_TO_EDGE);
  mLUT.SetWrapT(GL_CLAMP_TO_EDGE);
  mLUT.Unbind();

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

void Renderer::setData(std::shared_ptr<Image2D> const& image) {
  mHasTexture = image != nullptr;

  if (!mHasTexture) {
    return;
  }

  mBounds     = image->mBounds;
  mNumScalars = image->mNumScalars;

  if (std::holds_alternative<U8ValueVector>(image->mPoints)) {
    auto imageData = std::get<U8ValueVector>(image->mPoints);
    uploadTextureData<uint8_t>(mTexture, imageData, image);

  } else if (std::holds_alternative<U16ValueVector>(image->mPoints)) {
    auto imageData = std::get<U16ValueVector>(image->mPoints);
    uploadTextureData<uint16_t>(mTexture, imageData, image);

  } else if (std::holds_alternative<U32ValueVector>(image->mPoints)) {
    auto imageData = std::get<U32ValueVector>(image->mPoints);
    uploadTextureData<uint32_t>(mTexture, imageData, image);

  } else if (std::holds_alternative<I16ValueVector>(image->mPoints)) {
    auto imageData = std::get<I16ValueVector>(image->mPoints);
    uploadTextureData<int16_t>(mTexture, imageData, image);

  } else if (std::holds_alternative<I32ValueVector>(image->mPoints)) {
    auto imageData = std::get<I32ValueVector>(image->mPoints);
    uploadTextureData<int32_t>(mTexture, imageData, image);

  } else if (std::holds_alternative<F32ValueVector>(image->mPoints)) {
    auto imageData = std::get<F32ValueVector>(image->mPoints);
    uploadTextureData<float>(mTexture, imageData, image);

  } else {
    logger().error("Unknown type!");
  }
}

void Renderer::setLUT(std::vector<glm::vec4> const& lut) {
  mHasLUT = !lut.empty();

  if (mHasLUT) {
    mLUT.Bind();
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, static_cast<int32_t>(lut.size()), 0, GL_RGBA,
        GL_FLOAT, lut.data());
    glTexParameteri(mLUT.GetTarget(), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    mLUT.Unbind();
  }
}

void Renderer::setMinMax(glm::vec2 const& minMax) {
  mMinMax = minMax;
}

void Renderer::setOpacity(float opacity) {
  mOpacity = opacity;
}

float Renderer::getOpacity() const {
  return mOpacity;
}

void Renderer::setObject(std::string objectName) {
  mObjectName = std::move(objectName);
  if (mObjectName == "None" || mObjectName.empty()) {
    return;
  }

  auto object = mSolarSystem->getObject(mObjectName);
  if (!object) {
    return;
  }

  mMinBounds = -object->getRadii();
  mMaxBounds = object->getRadii();
}

std::string Renderer::getObject() const {
  return mObjectName;
}

bool Renderer::Do() {
  if (!mHasTexture || mObjectName == "None" || mObjectName.empty()) {
    return false;
  }

  auto object   = mSolarSystem->getObject(mObjectName);
  auto observer = mSolarSystem->getObserver();
  if (!object || object->getCenterName() != observer.getCenterName()) {
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

  if (mHasLUT) {
    mLUT.Bind(GL_TEXTURE3);
    mShader.SetUniform(mShader.GetUniformLocation("uLUT"), 3);
  }

  mShader.SetUniform(mShader.GetUniformLocation("uHasLUT"), mHasLUT);
  mShader.SetUniform(mShader.GetUniformLocation("uNumScalars"), static_cast<int>(mNumScalars));
  mShader.SetUniform(mShader.GetUniformLocation("uOpacity"), mOpacity);
  mShader.SetUniform(mShader.GetUniformLocation("uDepthBuffer"), 0);
  mShader.SetUniform(mShader.GetUniformLocation("uTexture"), 1);
  mShader.SetUniform(mShader.GetUniformLocation("uValueRange"), mMinMax.x, mMinMax.y);

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

  if (mHasLUT) {
    mLUT.Unbind(GL_TEXTURE3);
  }
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