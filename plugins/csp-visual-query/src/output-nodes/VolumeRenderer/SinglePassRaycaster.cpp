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

#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/type_ptr.hpp>

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
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eAtmospheres) + 10);
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

  if (mShaderDirty) {
    mShader = VistaGLSLShader();

    std::string shaderRoot = "../share/resources/shaders/csp-visual-query/";

    mShader.InitGeometryShaderFromString(
        cs::utils::filesystem::loadToString(shaderRoot + "SPRaycaster.geom.glsl"));
    mShader.InitVertexShaderFromString(
        cs::utils::filesystem::loadToString(shaderRoot + "SPRaycaster.vert.glsl"));
    mShader.InitFragmentShaderFromString(
        cs::utils::filesystem::loadToString(shaderRoot + "SPRaycaster.frag.glsl"));
    mShader.Link();

    mUniforms.texture     = mShader.GetUniformLocation("uTexture");
    mUniforms.matInvMV    = mShader.GetUniformLocation("uMatInvMV");
    mUniforms.matInvP     = mShader.GetUniformLocation("uMatInvP");
    mUniforms.lonRange    = mShader.GetUniformLocation("uLonRange");
    mUniforms.latRange    = mShader.GetUniformLocation("uLatRange");
    mUniforms.heightRange = mShader.GetUniformLocation("uHeightRange");
    mUniforms.innerRadii  = mShader.GetUniformLocation("uInnerRadii");
    mUniforms.outerRadii  = mShader.GetUniformLocation("uOuterRadii");
    mUniforms.bodyRadii   = mShader.GetUniformLocation("uBodyRadii");

    mShaderDirty = false;
  }

  glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);
  glEnable(GL_TEXTURE_3D);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  float scale     = mSettings->mGraphics.pHeightScale.get();
  auto  radii     = object->getRadii();
  auto  transform = object->getObserverRelativeTransform();

  glm::dmat4 matInverseEllipsoid(
      1, 0, 0, 0, 0, radii[0] / radii[1], 0, 0, 0, 0, radii[0] / radii[2], 0, 0, 0, 0, 1);

  // get matrices and related values
  std::array<GLfloat, 16> glMatV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matV = glm::make_mat4x4(glMatV.data());
  auto matP = glm::make_mat4x4(glMatP.data());

  glm::dmat4 matInvV     = glm::inverse(matV);
  glm::dmat4 matInvWorld = glm::inverse(transform);

  glm::mat4 matInvMV = matInverseEllipsoid * matInvWorld * matInvV;
  glm::mat4 matInvP  = glm::inverse(matP);

  mShader.Bind();
  mTexture.Bind(GL_TEXTURE0);

  mShader.SetUniform(mUniforms.texture, 0);

  glUniformMatrix4fv(mUniforms.matInvMV, 1, false, glm::value_ptr(matInvMV));
  glUniformMatrix4fv(mUniforms.matInvP, 1, false, glm::value_ptr(matInvP));

  glUniform3fv(mUniforms.bodyRadii, 1, glm::value_ptr(glm::vec3(radii)));

  double    toRad     = glm::pi<double>() / 180.0;
  glm::vec2 lonBounds = glm::dvec2(mBounds.mMinLon, mBounds.mMaxLon) * toRad;
  glm::vec2 latBounds = glm::dvec2(mBounds.mMinLat, mBounds.mMaxLat) * toRad;

  glm::dvec2 heightBounds =
      glm::dvec2(mBounds.mMinHeight, mBounds.mMaxHeight) * static_cast<double>(scale);

  glUniform2f(mUniforms.lonRange, lonBounds.x, lonBounds.y);
  glUniform2f(mUniforms.latRange, latBounds.x, latBounds.y);
  glUniform2f(mUniforms.heightRange, static_cast<float>(heightBounds.x), static_cast<float>(heightBounds.y));

  glm::vec3 innerRadii = radii + heightBounds.x;
  glUniform3fv(mUniforms.innerRadii, 1, glm::value_ptr(innerRadii));

  glm::vec3 outerRadii = radii + heightBounds.y;
  glUniform3fv(mUniforms.outerRadii, 1, glm::value_ptr(outerRadii));

  glDrawArrays(GL_POINTS, 0, 1);

  mTexture.Unbind();
  mShader.Release();

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glPopAttrib();

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
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, image->mDimension.x, image->mDimension.y,
        image->mDimension.z, 0, imageFormat, GL_FLOAT, data.data());

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