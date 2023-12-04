////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SinglePassRaycaster.hpp"

#include "../../../../../src/cs-core/Settings.hpp"
#include "../../../../../src/cs-core/SolarSystem.hpp"
#include "../../../../../src/cs-utils/FrameStats.hpp"
#include "../../../../../src/cs-utils/utils.hpp"
#include "../../../../../src/cs-utils/filesystem.hpp"
#include "../../logger.hpp"

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
    , mPreLookupTexture(GL_TEXTURE_2D)
    , mHasTexture(false)
    , mMinBounds(0)
    , mMaxBounds(0) {

  mTexture.Bind();
  mTexture.SetWrapS(GL_CLAMP_TO_EDGE);
  mTexture.SetWrapT(GL_CLAMP_TO_EDGE);
  mTexture.Unbind();

  mPreLookupTexture.Bind();
  mPreLookupTexture.SetWrapS(GL_CLAMP_TO_EDGE);
  mPreLookupTexture.SetWrapT(GL_CLAMP_TO_EDGE);
  mPreLookupTexture.Unbind();

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

  cs::utils::FrameStats::ScopedTimer timer("VolumeRenderer");

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

    mUniforms.texture          = mShader.GetUniformLocation("uTexture");
    mUniforms.preLookupTexture = mShader.GetUniformLocation("uPreLookupTexture");
    mUniforms.matInvMV         = mShader.GetUniformLocation("uMatInvMV");
    mUniforms.matInvP          = mShader.GetUniformLocation("uMatInvP");
    mUniforms.lonRange         = mShader.GetUniformLocation("uLonRange");
    mUniforms.latRange         = mShader.GetUniformLocation("uLatRange");
    mUniforms.heightRange      = mShader.GetUniformLocation("uHeightRange");
    mUniforms.innerRadii       = mShader.GetUniformLocation("uInnerRadii");
    mUniforms.outerRadii       = mShader.GetUniformLocation("uOuterRadii");
    mUniforms.bodyRadii        = mShader.GetUniformLocation("uBodyRadii");

    mShaderDirty = false;
  }

  glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);
  glEnable(GL_TEXTURE_3D);
  glEnable(GL_TEXTURE_2D);
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
  mPreLookupTexture.Bind(GL_TEXTURE1);

  mShader.SetUniform(mUniforms.texture, 0);
  mShader.SetUniform(mUniforms.preLookupTexture, 1);

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
  glUniform2f(mUniforms.heightRange, static_cast<float>(heightBounds.x),
      static_cast<float>(heightBounds.y));

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

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

/// This function checks if any cell in a subvolume has a non-transperent value.
template <typename T>
bool hasCellValue(std::shared_ptr<Volume3D> const& volume, glm::dvec3 start, glm::dvec3 end) {
  for (auto dz = static_cast<uint32_t>(start.z); dz < end.z; ++dz) {
    for (auto dy = static_cast<uint32_t>(start.y); dy < end.y; ++dy) {
      for (auto dx = static_cast<uint32_t>(start.x); dx < end.x; ++dx) {
        auto data = volume->at<T>(dx, dy, dz);
        // Check for transparency.
        if (data[3] != 0) {
          return true;
        }
      }
    }
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, GLenum GLType>
void SinglePassRaycaster::uploadVolume(std::shared_ptr<Volume3D> const& volume) {
  auto   imageData   = std::get<T>(volume->mPoints);
  GLenum imageFormat = get3DPixelFormat(volume->mNumScalars);

  auto data = copyTo3DTextureBuffer(imageData, volume->mNumScalars);

  mTexture.Bind();
  glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, volume->mDimension.x, volume->mDimension.y,
      volume->mDimension.z, 0, imageFormat, GLType, data.data());

  glTexParameteri(mTexture.GetTarget(), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  mTexture.Unbind();

  constexpr uint32_t xyExtends = 256;
  constexpr uint32_t zExtends  = sizeof(uint8_t) * 8;

  uint32_t              threads = std::thread::hardware_concurrency();
  cs::utils::ThreadPool tp(threads);
  uint32_t              taskSize = (xyExtends / threads) + 1;

  std::vector<uint8_t> data256{};
  data256.resize(xyExtends * xyExtends);
  std::fill(data256.begin(), data256.end(), 0);

  double stepZ = volume->mDimension.z / static_cast<double>(zExtends);
  double stepY = volume->mDimension.y / static_cast<double>(xyExtends);
  double stepX = volume->mDimension.x / static_cast<double>(xyExtends);

  for (uint32_t t = 0; t < threads; ++t) {
    tp.enqueue([=, &data256] {
      for (uint32_t ay = t * taskSize; ay < t * taskSize + taskSize && ay < xyExtends; ++ay) {
        double startY = ay * stepY;
        double endY   = startY + stepY;
        startY        = std::max(0.0, startY - stepY / 2.0);
        endY          = std::min(static_cast<double>(volume->mDimension.y), endY + stepY / 2.0);

        for (uint32_t ax = 0; ax < xyExtends; ++ax) {
          double startX = ax * stepX;
          double endX   = startX + stepX;
          startX        = std::max(0.0, startX - stepX / 2.0);
          endX          = std::min(static_cast<double>(volume->mDimension.x), endX + stepX / 2.0);

          for (uint32_t az = 0; az < zExtends; ++az) {
            double startZ = az * stepZ;
            double endZ   = startZ + stepZ;
            startZ        = std::max(0.0, startZ - stepZ / 2.0);
            endZ          = std::min(static_cast<double>(volume->mDimension.z), endZ + stepZ / 2.0);

            if (hasCellValue<T>(volume, {startX, startY, startZ}, {endX, endY, endZ})) {
              data256[ay * xyExtends + ax] |= static_cast<uint8_t>(std::pow(2, az));
            }
          }
        }
      }
    });
  }

  while (!tp.hasFinished()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  mPreLookupTexture.Bind();
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, xyExtends, xyExtends, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE,
      data256.data());
  glTexParameteri(mPreLookupTexture.GetTarget(), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  mPreLookupTexture.Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SinglePassRaycaster::setData(std::shared_ptr<Volume3D> const& image) {
  mHasTexture = image != nullptr;

  if (!mHasTexture) {
    return;
  }

  mBounds = image->mBounds;

  switch (image->mPoints.index()) {
  case cs::utils::variantIndex<PointsType, U8ValueVector>():
    uploadVolume<U8ValueVector, GL_UNSIGNED_BYTE>(image);
    break;

  case cs::utils::variantIndex<PointsType, U16ValueVector>():
    uploadVolume<U16ValueVector, GL_UNSIGNED_SHORT>(image);
    break;

  case cs::utils::variantIndex<PointsType, U32ValueVector>():
    uploadVolume<U32ValueVector, GL_UNSIGNED_INT>(image);
    break;

  case cs::utils::variantIndex<PointsType, I16ValueVector>():
    uploadVolume<I16ValueVector, GL_SHORT>(image);
    break;

  case cs::utils::variantIndex<PointsType, I32ValueVector>():
    uploadVolume<I32ValueVector, GL_INT>(image);
    break;

  case cs::utils::variantIndex<PointsType, F32ValueVector>():
    uploadVolume<F32ValueVector, GL_FLOAT>(image);
    break;

  default:
    logger().error("Unknown type!");
  }
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