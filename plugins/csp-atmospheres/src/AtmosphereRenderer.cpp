////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Atmosphere.hpp"

#ifdef _WIN32
#include <Windows.h>
#endif

#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/Shadows.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/GraphicsManager/VistaGeometryFactory.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/Rendering/VistaGeometryData.h>
#include <VistaOGLExt/Rendering/VistaGeometryRenderingCore.h>
#include <VistaOGLExt/VistaOGLUtils.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaTools/tinyXML/tinyxml.h>

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace csp::atmospheres {

AtmosphereRenderer::AtmosphereRenderer(std::shared_ptr<Plugin::Settings> settings)
    : mPluginSettings(std::move(settings)) {

  initData();

  // scene-wide settings -----------------------------------------------------
  mPluginSettings->mQuality.connectAndTouch([this](int val) { setPrimaryRaySteps(val); });

  mPluginSettings->mEnableWater.connectAndTouch([this](bool val) { setDrawWater(val); });

  mPluginSettings->mEnableClouds.connectAndTouch([this](bool val) {
    if (mUseClouds != val) {
      mShaderDirty = true;
      mUseClouds   = val;
    }
  });

  mPluginSettings->mWaterLevel.connectAndTouch([this](float val) { setWaterLevel(val / 1000); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setSun(glm::vec3 const& direction, float illuminance) {
  mSunIntensity = illuminance;
  mSunDirection = direction;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setRadii(glm::dvec3 const& radii) {
  mRadii = radii;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setWorldTransform(glm::dmat4 const& transform) {
  mWorldTransform = transform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setClouds(std::string const& textureFile, float height) {

  if (mCloudTextureFile != textureFile) {
    mCloudTextureFile = textureFile;
    mCloudTexture.reset();
    if (!textureFile.empty()) {
      mCloudTexture = cs::graphics::TextureLoader::loadFromFile(textureFile);
    }
    mShaderDirty = true;
    mUseClouds   = mCloudTexture != nullptr;
  }

  mCloudHeight = height;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setShadowMap(std::shared_ptr<cs::graphics::ShadowMap> const& pShadowMap) {
  if (mShadowMap != pShadowMap) {
    mShadowMap   = pShadowMap;
    mShaderDirty = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setHDRBuffer(std::shared_ptr<cs::graphics::HDRBuffer> const& pHDRBuffer) {
  if (mHDRBuffer != pHDRBuffer) {
    mHDRBuffer   = pHDRBuffer;
    mShaderDirty = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float AtmosphereRenderer::getApproximateSceneBrightness() const {
  return mApproximateBrightness;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int AtmosphereRenderer::getPrimaryRaySteps() const {
  return mPrimaryRaySteps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setPrimaryRaySteps(int iValue) {
  mPrimaryRaySteps = iValue;
  mShaderDirty     = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int AtmosphereRenderer::getSecondaryRaySteps() const {
  return mSecondaryRaySteps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setSecondaryRaySteps(int iValue) {
  mSecondaryRaySteps = iValue;
  mShaderDirty       = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float AtmosphereRenderer::getAtmosphereHeight() const {
  return mAtmosphereHeight;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setAtmosphereHeight(float dValue) {
  mAtmosphereHeight = dValue;
  mShaderDirty      = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float AtmosphereRenderer::getMieHeight() const {
  return mMieHeight;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setMieHeight(float dValue) {
  mMieHeight   = dValue;
  mShaderDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec3 AtmosphereRenderer::getMieScattering() const {
  return mMieScattering;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setMieScattering(const glm::vec3& vValue) {
  mMieScattering = vValue;
  mShaderDirty   = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float AtmosphereRenderer::getMieAnisotropy() const {
  return mMieAnisotropy;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setMieAnisotropy(float dValue) {
  mMieAnisotropy = dValue;
  mShaderDirty   = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float AtmosphereRenderer::getRayleighHeight() const {
  return mRayleighHeight;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setRayleighHeight(float dValue) {
  mRayleighHeight = dValue;
  mShaderDirty    = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec3 AtmosphereRenderer::getRayleighScattering() const {
  return mRayleighScattering;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setRayleighScattering(const glm::vec3& vValue) {
  mRayleighScattering = vValue;
  mShaderDirty        = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float AtmosphereRenderer::getRayleighAnisotropy() const {
  return mRayleighAnisotropy;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setRayleighAnisotropy(float dValue) {
  mRayleighAnisotropy = dValue;
  mShaderDirty        = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AtmosphereRenderer::getDrawSun() const {
  return mDrawSun;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setDrawSun(bool bEnable) {
  mDrawSun     = bEnable;
  mShaderDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AtmosphereRenderer::getDrawWater() const {
  return mDrawWater;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setDrawWater(bool bEnable) {
  mDrawWater   = bEnable;
  mShaderDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float AtmosphereRenderer::getWaterLevel() const {
  return mWaterLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setWaterLevel(float fValue) {
  mWaterLevel = fValue;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float AtmosphereRenderer::getAmbientBrightness() const {
  return mAmbientBrightness;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setAmbientBrightness(float fValue) {
  mAmbientBrightness = fValue;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AtmosphereRenderer::getUseToneMapping() const {
  return mUseToneMapping;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setUseToneMapping(bool bEnable, float fExposure, float fGamma) {
  mUseToneMapping = bEnable;
  mExposure       = fExposure;
  mGamma          = fGamma;
  mShaderDirty    = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AtmosphereRenderer::getUseLinearDepthBuffer() const {
  return mUseLinearDepthBuffer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setUseLinearDepthBuffer(bool bEnable) {
  mUseLinearDepthBuffer = bEnable;
  mShaderDirty          = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::updateShader() {
  mAtmoShader = VistaGLSLShader();

  std::string sVert(cAtmosphereVert);
  std::string sFrag(cAtmosphereFrag0);
  sFrag.append(cAtmosphereFrag1);

  cs::utils::replaceString(sFrag, "HEIGHT_ATMO", cs::utils::toString(mAtmosphereHeight));
  cs::utils::replaceString(sFrag, "ANISOTROPY_R", cs::utils::toString(mRayleighAnisotropy));
  cs::utils::replaceString(sFrag, "ANISOTROPY_M", cs::utils::toString(mMieAnisotropy));
  cs::utils::replaceString(sFrag, "HEIGHT_R", cs::utils::toString(mRayleighHeight));
  cs::utils::replaceString(sFrag, "HEIGHT_M", cs::utils::toString(mMieHeight));
  cs::utils::replaceString(sFrag, "BETA_R_0", cs::utils::toString(mRayleighScattering[0]));
  cs::utils::replaceString(sFrag, "BETA_R_1", cs::utils::toString(mRayleighScattering[1]));
  cs::utils::replaceString(sFrag, "BETA_R_2", cs::utils::toString(mRayleighScattering[2]));
  cs::utils::replaceString(sFrag, "BETA_M_0", cs::utils::toString(mMieScattering[0]));
  cs::utils::replaceString(sFrag, "BETA_M_1", cs::utils::toString(mMieScattering[1]));
  cs::utils::replaceString(sFrag, "BETA_M_2", cs::utils::toString(mMieScattering[2]));
  cs::utils::replaceString(sFrag, "PRIMARY_RAY_STEPS", cs::utils::toString(mPrimaryRaySteps));
  cs::utils::replaceString(sFrag, "SECONDARY_RAY_STEPS", cs::utils::toString(mSecondaryRaySteps));
  cs::utils::replaceString(sFrag, "ENABLE_TONEMAPPING", std::to_string(mUseToneMapping));
  cs::utils::replaceString(sFrag, "EXPOSURE", cs::utils::toString(mExposure));
  cs::utils::replaceString(sFrag, "GAMMA", cs::utils::toString(mGamma));
  cs::utils::replaceString(sFrag, "HEIGHT_ATMO", cs::utils::toString(mAtmosphereHeight));
  cs::utils::replaceString(sFrag, "USE_LINEARDEPTHBUFFER", std::to_string(mUseLinearDepthBuffer));
  cs::utils::replaceString(sFrag, "DRAW_SUN", std::to_string(mDrawSun));
  cs::utils::replaceString(sFrag, "DRAW_WATER", std::to_string(mDrawWater));
  cs::utils::replaceString(sFrag, "USE_SHADOWMAP", std::to_string(mShadowMap != nullptr));
  cs::utils::replaceString(sFrag, "USE_CLOUDMAP", std::to_string(mUseClouds && mCloudTexture));
  cs::utils::replaceString(sFrag, "ENABLE_HDR", std::to_string(mHDRBuffer != nullptr));
  cs::utils::replaceString(sFrag, "HDR_SAMPLES",
      mHDRBuffer == nullptr ? "0" : std::to_string(mHDRBuffer->getMultiSamples()));

  mAtmoShader.InitVertexShaderFromString(sVert);
  mAtmoShader.InitFragmentShaderFromString(sFrag);

  mAtmoShader.Link();

  mUniforms.sunIntensity      = mAtmoShader.GetUniformLocation("uSunIntensity");
  mUniforms.sunDir            = mAtmoShader.GetUniformLocation("uSunDir");
  mUniforms.farClip           = mAtmoShader.GetUniformLocation("uFarClip");
  mUniforms.waterLevel        = mAtmoShader.GetUniformLocation("uWaterLevel");
  mUniforms.ambientBrightness = mAtmoShader.GetUniformLocation("uAmbientBrightness");
  mUniforms.depthBuffer       = mAtmoShader.GetUniformLocation("uDepthBuffer");
  mUniforms.colorBuffer       = mAtmoShader.GetUniformLocation("uColorBuffer");
  mUniforms.cloudTexture      = mAtmoShader.GetUniformLocation("uCloudTexture");
  mUniforms.cloudAltitude     = mAtmoShader.GetUniformLocation("uCloudAltitude");
  mUniforms.shadowCascades    = mAtmoShader.GetUniformLocation("uShadowCascades");

  for (size_t i = 0; i < 5; ++i) {
    mUniforms.shadowMaps.at(i) = glGetUniformLocation(
        mAtmoShader.GetProgram(), ("uShadowMaps[" + std::to_string(i) + "]").c_str());
    mUniforms.shadowProjectionMatrices.at(i) = glGetUniformLocation(mAtmoShader.GetProgram(),
        ("uShadowProjectionViewMatrices[" + std::to_string(i) + "]").c_str());
  }

  mUniforms.inverseModelViewMatrix           = mAtmoShader.GetUniformLocation("uMatInvMV");
  mUniforms.inverseModelViewProjectionMatrix = mAtmoShader.GetUniformLocation("uMatInvMVP");
  mUniforms.inverseProjectionMatrix          = mAtmoShader.GetUniformLocation("uMatInvP");
  mUniforms.modelViewMatrix                  = mAtmoShader.GetUniformLocation("uMatMV");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AtmosphereRenderer::Do() {
  cs::utils::FrameTimings::ScopedTimer timer("Render Atmosphere");

  if (mShaderDirty) {
    updateShader();
    mShaderDirty = false;
  }

  // save current lighting and meterial state of the OpenGL state machine ----
  glPushAttrib(GL_LIGHTING_BIT | GL_ENABLE_BIT);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glEnable(GL_CULL_FACE);
  glEnable(GL_TEXTURE_2D);
  glCullFace(GL_FRONT);
  glDepthMask(GL_FALSE);

  // copy depth buffer -------------------------------------------------------
  if (!mHDRBuffer) {
    std::array<GLint, 4> iViewport{};
    glGetIntegerv(GL_VIEWPORT, iViewport.data());

    auto* viewport   = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
    auto const& data = mGBufferData[viewport];

    data.mDepthBuffer->Bind();
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, iViewport.at(0), iViewport.at(1),
        iViewport.at(2), iViewport.at(3), 0);
    data.mColorBuffer->Bind();
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, iViewport.at(0), iViewport.at(1), iViewport.at(2),
        iViewport.at(3), 0);
  }

  // get matrices and related values -----------------------------------------
  std::array<GLfloat, 16> glMatMV{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glm::mat4 matMV(glm::make_mat4x4(glMatMV.data()) * glm::mat4(mWorldTransform) *
                  glm::mat4(static_cast<float>(mRadii[0] / (1.0 - mAtmosphereHeight)), 0, 0, 0, 0,
                      static_cast<float>(mRadii[1] / (1.0 - mAtmosphereHeight)), 0, 0, 0, 0,
                      static_cast<float>(mRadii[2] / (1.0 - mAtmosphereHeight)), 0, 0, 0, 0, 1));

  auto matInvMV = glm::inverse(matMV);

  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  glm::mat4 matInvP = glm::inverse(glm::make_mat4x4(glMatP.data()));
  glm::mat4 matInvMVP(matInvMV * matInvP);

  glm::vec3 sunDir =
      glm::normalize(glm::vec3(glm::inverse(mWorldTransform) * glm::vec4(mSunDirection, 0)));

  // set uniforms ------------------------------------------------------------
  mAtmoShader.Bind();

  mAtmoShader.SetUniform(mUniforms.sunIntensity, mSunIntensity);
  mAtmoShader.SetUniform(mUniforms.sunDir, sunDir[0], sunDir[1], sunDir[2]);
  mAtmoShader.SetUniform(mUniforms.farClip, cs::utils::getCurrentFarClipDistance());

  mAtmoShader.SetUniform(mUniforms.waterLevel, mWaterLevel);
  mAtmoShader.SetUniform(mUniforms.ambientBrightness, mAmbientBrightness);

  if (mHDRBuffer) {
    mHDRBuffer->doPingPong();
    mHDRBuffer->bind();
    mHDRBuffer->getDepthAttachment()->Bind(GL_TEXTURE0);
    mHDRBuffer->getCurrentReadAttachment()->Bind(GL_TEXTURE1);
  } else {
    auto* viewport   = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
    auto const& data = mGBufferData[viewport];
    data.mDepthBuffer->Bind(GL_TEXTURE0);
    data.mColorBuffer->Bind(GL_TEXTURE1);
  }

  mAtmoShader.SetUniform(mUniforms.depthBuffer, 0);
  mAtmoShader.SetUniform(mUniforms.colorBuffer, 1);

  if (mUseClouds && mCloudTexture) {
    mCloudTexture->Bind(GL_TEXTURE3);
    mAtmoShader.SetUniform(mUniforms.cloudTexture, 3);
    mAtmoShader.SetUniform(mUniforms.cloudAltitude, mCloudHeight);
  }

  if (mShadowMap) {
    int texUnitShadow = 4;
    mAtmoShader.SetUniform(
        mUniforms.shadowCascades, static_cast<int>(mShadowMap->getMaps().size()));
    for (size_t i = 0; i < mShadowMap->getMaps().size(); ++i) {
      mShadowMap->getMaps()[i]->Bind(
          static_cast<GLenum>(GL_TEXTURE0) + texUnitShadow + static_cast<int>(i));
      glUniform1i(mUniforms.shadowMaps.at(i), texUnitShadow + static_cast<int>(i));

      auto mat = mShadowMap->getShadowMatrices()[i];
      glUniformMatrix4fv(mUniforms.shadowProjectionMatrices.at(i), 1, GL_FALSE, mat.GetData());
    }
  }

  // Why is there no set uniform for matrices???
  glUniformMatrix4fv(mUniforms.inverseModelViewMatrix, 1, GL_FALSE, glm::value_ptr(matInvMV));
  glUniformMatrix4fv(
      mUniforms.inverseModelViewProjectionMatrix, 1, GL_FALSE, glm::value_ptr(matInvMVP));
  glUniformMatrix4fv(mUniforms.inverseProjectionMatrix, 1, GL_FALSE, glm::value_ptr(matInvP));
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(matMV));

  // draw --------------------------------------------------------------------
  mQuadVAO.Bind();
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  mQuadVAO.Release();

  // clean up ----------------------------------------------------------------

  if (mHDRBuffer) {
    mHDRBuffer->getDepthAttachment()->Unbind(GL_TEXTURE0);
    mHDRBuffer->getCurrentReadAttachment()->Unbind(GL_TEXTURE1);
  } else {
    auto* viewport   = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
    auto const& data = mGBufferData[viewport];
    data.mDepthBuffer->Unbind(GL_TEXTURE0);
    data.mColorBuffer->Unbind(GL_TEXTURE1);
  }

  if (mUseClouds && mCloudTexture) {
    mCloudTexture->Unbind(GL_TEXTURE3);
  }

  mAtmoShader.Release();

  glDepthMask(GL_TRUE);

  glPopAttrib();

  // update brightness value -------------------------------------------------
  // This is a crude approximation of the overall scene brightness due to
  // atmospheric scattering, camera position and the sun's position.
  // It may be used for fake HDR effects such as dimming stars.

  // some required positions and directions
  glm::vec4 temp            = matInvMVP * glm::vec4(0, 0, 0, 1);
  glm::vec3 vCamera         = glm::vec3(temp) / temp[3];
  glm::vec3 vPlanet         = glm::vec3(0, 0, 0);
  glm::vec3 vCameraToPlanet = glm::normalize(vCamera - vPlanet);

  // [planet surface ... 5x atmosphere boundary] -> [0 ... 1]
  float fHeightInAtmosphere = std::min(1.0F,
      std::max(0.0F, (glm::length(vCamera) - (1.F - mAtmosphereHeight)) / (mAtmosphereHeight * 5)));

  // [noon ... midnight] -> [1 ... -1]
  float fDaySide = glm::dot(vCameraToPlanet, sunDir);

  // limit brightness when on night side (also in dusk an dawn time)
  float const exponent       = 50.F;
  float fBrightnessOnSurface = std::pow(std::min(1.F, std::max(0.F, fDaySide + 1.F)), exponent);

  // reduce brightness in outer space
  mApproximateBrightness = (1.F - fHeightInAtmosphere) * fBrightnessOnSurface;

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AtmosphereRenderer::GetBoundingBox(VistaBoundingBox& bb) {
  auto const extend = static_cast<float>(1.0 / (1.0 - mAtmosphereHeight));

  // Boundingbox is computed by translation an edge points
  std::array<float, 3> const fMin = {-extend, -extend, -extend};
  std::array<float, 3> const fMax = {extend, extend, extend};

  bb.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::initData() {
  // create quad -------------------------------------------------------------
  std::array<float, 8> const data{-1, 1, 1, 1, -1, -1, 1, -1};

  mQuadVBO.Bind(GL_ARRAY_BUFFER);
  mQuadVBO.BufferData(data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);
  mQuadVBO.Release();

  // positions
  mQuadVAO.EnableAttributeArray(0);
  mQuadVAO.SpecifyAttributeArrayFloat(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0, &mQuadVBO);

  // create textures ---------------------------------------------------------
  for (auto const& viewport : GetVistaSystem()->GetDisplayManager()->GetViewports()) {
    GBufferData bufferData;

    bufferData.mDepthBuffer = std::make_unique<VistaTexture>(GL_TEXTURE_2D);
    bufferData.mDepthBuffer->Bind();
    bufferData.mDepthBuffer->SetWrapS(GL_CLAMP);
    bufferData.mDepthBuffer->SetWrapT(GL_CLAMP);
    bufferData.mDepthBuffer->SetMinFilter(GL_NEAREST);
    bufferData.mDepthBuffer->SetMagFilter(GL_NEAREST);
    bufferData.mDepthBuffer->Unbind();

    bufferData.mColorBuffer = std::make_unique<VistaTexture>(GL_TEXTURE_2D);
    bufferData.mColorBuffer->Bind();
    bufferData.mColorBuffer->SetWrapS(GL_CLAMP);
    bufferData.mColorBuffer->SetWrapT(GL_CLAMP);
    bufferData.mColorBuffer->SetMinFilter(GL_NEAREST);
    bufferData.mColorBuffer->SetMagFilter(GL_NEAREST);
    bufferData.mColorBuffer->Unbind();

    mGBufferData.emplace(viewport.second, std::move(bufferData));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres
