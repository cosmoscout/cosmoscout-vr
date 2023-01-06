////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Atmosphere.hpp"

#include "ModelBase.hpp"
#include "logger.hpp"
#include "models/bruneton/Model.hpp"
#include "models/cosmoscout/Model.hpp"

#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "../../../src/cs-utils/filesystem.hpp"
// #include "../../../src/cs-core/EclipseShadowReceiver.hpp"
// #include "../../../src/cs-graphics/Shadows.hpp"
// #include "../../../src/cs-graphics/TextureLoader.hpp"
// #include "../../../src/cs-utils/utils.hpp"
// #include "../../../src/cs-scene/CelestialObject.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaOGLExt/VistaTexture.h>
// #include <VistaBase/VistaVectorMath.h>
// #include <VistaKernel/DisplayManager/VistaViewport.h>
// #include <VistaKernel/GraphicsManager/VistaGeometryFactory.h>
// #include <VistaMath/VistaBoundingBox.h>
// #include <VistaOGLExt/Rendering/VistaGeometryData.h>
// #include <VistaOGLExt/Rendering/VistaGeometryRenderingCore.h>
// #include <VistaOGLExt/VistaOGLUtils.h>
// #include <VistaTools/tinyXML/tinyxml.h>

#include <glm/gtc/type_ptr.hpp>
// #include <glm/glm.hpp>
// #include <memory>
// #include <unordered_map>
// #include <utility>

namespace csp::atmospheres {

////////////////////////////////////////////////////////////////////////////////////////////////////

Atmosphere::Atmosphere(std::shared_ptr<Plugin::Settings> pluginSettings,
    std::shared_ptr<cs::core::Settings>                  allSettings,
    std::shared_ptr<cs::core::SolarSystem>               solarSystem,
    std::shared_ptr<cs::core::GraphicsEngine> graphicsEngine, std::string objectName)
    : mPluginSettings(std::move(pluginSettings))
    , mAllSettings(std::move(allSettings))
    , mSolarSystem(std::move(solarSystem))
    , mGraphicsEngine(std::move(graphicsEngine))
    , mObjectName(std::move(objectName)) {
    // , mEclipseShadowReceiver(
    //       std::make_shared<cs::core::EclipseShadowReceiver>(mAllSettings, mSolarSystem, false)) {

  // mEnableShadowsConnection = mAllSettings->mGraphics.pEnableShadows.connectAndTouch([this](bool
  // value) {
  //   for (auto const& atmosphere : mAtmospheres) {
  //     if (value && mPluginSettings->mEnableLightShafts.get()) {
  //       atmosphere.second->getRenderer().setShadowMap(mGraphicsEngine->getShadowMap());
  //     } else {
  //       atmosphere.second->getRenderer().setShadowMap(nullptr);
  //     }
  //   }
  // });

  mEnableHDRConnection = mAllSettings->mGraphics.pEnableHDR.connectAndTouch([this](bool val) {
    if (val && !mHDRBuffer) {
      mHDRBuffer   = mGraphicsEngine->getHDRBuffer();
      mShaderDirty = true;
    } else if (!val && mHDRBuffer) {
      mHDRBuffer   = nullptr;
      mShaderDirty = true;
    }
  });

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mAtmosphereNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  mAtmosphereNode->SetIsEnabled(false);
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mAtmosphereNode.get(), static_cast<int>(cs::utils::DrawOrder::eAtmospheres));

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

Atmosphere::~Atmosphere() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mAtmosphereNode.get());

  mAllSettings->mGraphics.pEnableShadows.disconnect(mEnableShadowsConnection);
  mAllSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Atmosphere::configure(Plugin::Settings::Atmosphere const& settings) {
  auto object = mSolarSystem->getObject(mObjectName);
  if (object) {
    auto radii = object->getRadii();

    // TODO: Recreate model if type changed.
    if (!mModel) {
      switch (settings.mModel.get()) {
      case Plugin::Settings::Atmosphere::Model::eCosmoScoutVR:
        mModel = std::make_unique<models::cosmoscout::Model>();
        break;
      case Plugin::Settings::Atmosphere::Model::eBruneton:
        mModel = std::make_unique<models::bruneton::Model>();
        break;
      }
    }

    if (mModel->init(settings.mModelSettings, radii[0], radii[0] + settings.mHeight)) {
      mShaderDirty = true;
    }

    if (mRadii != radii) {
      mRadii       = radii;
      mShaderDirty = true;
    }

    if (mSettings != settings) {
      mSettings    = settings;
      mShaderDirty = true;
    }

    //     if (mCloudTextureFile != textureFile) {
    //   mCloudTextureFile = textureFile;
    //   mCloudTexture.reset();
    //   if (!textureFile.empty()) {
    //     mCloudTexture = cs::graphics::TextureLoader::loadFromFile(textureFile);
    //   }
    //   mShaderDirty = true;
    //   mUseClouds   = mCloudTexture != nullptr;
    // }

    // mCloudHeight = height;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Atmosphere::updateShader() {
  mAtmoShader = VistaGLSLShader();

  auto sVert = cs::utils::filesystem::loadToString(
      "../share/resources/shaders/csp-atmospheres/atmosphere.vert");
  auto sFrag = cs::utils::filesystem::loadToString(
      "../share/resources/shaders/csp-atmospheres/atmosphere.frag");

  cs::utils::replaceString(sFrag, "PLANET_RADIUS", std::to_string(mRadii[0]));
  cs::utils::replaceString(
      sFrag, "ATMOSPHERE_RADIUS", std::to_string(mRadii[0] + mSettings.mHeight));
  // cs::utils::replaceString(sFrag, "USE_SHADOWMAP", std::to_string(mShadowMap != nullptr));
  // cs::utils::replaceString(sFrag, "USE_CLOUDMAP", std::to_string(mUseClouds && mCloudTexture));
  cs::utils::replaceString(sFrag, "ENABLE_HDR", std::to_string(mHDRBuffer != nullptr));
  cs::utils::replaceString(sFrag, "HDR_SAMPLES",
      mHDRBuffer == nullptr ? "0" : std::to_string(mHDRBuffer->getMultiSamples()));

  // If the atmosphere should receive eclipse shadows, we need to inject the corresponding shader
  // source code snippet. If no eclipse shadow receiver was given, we just add a dummy method.
  // if (mEclipseShadowReceiver) {
  //   cs::utils::replaceString(
  //       sFrag, "ECLIPSE_SHADER_SNIPPET", mEclipseShadowReceiver->getShaderSnippet());
  // } else {
  //   cs::utils::replaceString(sFrag, "ECLIPSE_SHADER_SNIPPET",
  //       "vec3 getEclipseShadow(vec3 position) { return vec3(1); }");
  // }

  mAtmoShader.InitVertexShaderFromString(sVert);
  mAtmoShader.InitFragmentShaderFromString(sFrag);

  glAttachShader(mAtmoShader.GetProgram(), mModel->getShader());

  mAtmoShader.Link();

  mUniforms.sunDir         = mAtmoShader.GetUniformLocation("uSunDir");
  mUniforms.sunIlluminance = mAtmoShader.GetUniformLocation("uSunIlluminance");
  mUniforms.depthBuffer    = mAtmoShader.GetUniformLocation("uDepthBuffer");
  mUniforms.colorBuffer    = mAtmoShader.GetUniformLocation("uColorBuffer");
  // mUniforms.cloudTexture   = mAtmoShader.GetUniformLocation("uCloudTexture");
  // mUniforms.cloudAltitude  = mAtmoShader.GetUniformLocation("uCloudAltitude");
  // mUniforms.shadowCascades = mAtmoShader.GetUniformLocation("uShadowCascades");

  // for (size_t i = 0; i < 5; ++i) {
  //   mUniforms.shadowMaps.at(i) = glGetUniformLocation(
  //       mAtmoShader.GetProgram(), ("uShadowMaps[" + std::to_string(i) + "]").c_str());
  //   mUniforms.shadowProjectionMatrices.at(i) = glGetUniformLocation(mAtmoShader.GetProgram(),
  //       ("uShadowProjectionViewMatrices[" + std::to_string(i) + "]").c_str());
  // }

  mUniforms.inverseModelViewMatrix           = mAtmoShader.GetUniformLocation("uMatInvMV");
  mUniforms.inverseModelViewProjectionMatrix = mAtmoShader.GetUniformLocation("uMatInvMVP");
  mUniforms.inverseProjectionMatrix          = mAtmoShader.GetUniformLocation("uMatInvP");
  // mUniforms.modelViewMatrix                  = mAtmoShader.GetUniformLocation("uMatMV");
  // mUniforms.modelMatrix                      = mAtmoShader.GetUniformLocation("uMatM");

  // We bind the eclipse shadow map to texture unit 4.
  // mEclipseShadowReceiver->init(&mAtmoShader, 4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Atmosphere::update() {
  auto object = mSolarSystem->getObject(mObjectName);

  if (object && object->getIsBodyVisible() && mPluginSettings->mEnable.get()) {
    mSunIlluminance = mSolarSystem->getSunIlluminance(object->getObserverRelativePosition());
    mSunDirection   = mSolarSystem->getSunDirection(object->getObserverRelativePosition());
    mWorldTransform = object->getObserverRelativeTransform();
    // mEclipseShadowReceiver->update(*object);

    mAtmosphereNode->SetIsEnabled(true);
  } else {
    mAtmosphereNode->SetIsEnabled(false);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Atmosphere::Do() {
  cs::utils::FrameTimings::ScopedTimer timer("Render Atmosphere");

  if (mShaderDirty /*|| mEclipseShadowReceiver->needsRecompilation()*/) {
    updateShader();
    mShaderDirty = false;
  }

  // save current lighting and meterial state of the OpenGL state machine ----
  glPushAttrib(GL_LIGHTING_BIT | GL_ENABLE_BIT);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glEnable(GL_TEXTURE_2D);
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
  glm::mat4 matM(
      glm::mat4(mWorldTransform) * glm::mat4(static_cast<float>(mRadii[0] / mRadii[0]), 0, 0, 0, 0,
                                       static_cast<float>(mRadii[1] / mRadii[0]), 0, 0, 0, 0,
                                       static_cast<float>(mRadii[2] / mRadii[0]), 0, 0, 0, 0, 1));

  std::array<GLfloat, 16> glMatMV{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glm::mat4 matMV(glm::make_mat4x4(glMatMV.data()) * matM);

  auto matInvMV = glm::inverse(matMV);

  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  glm::mat4 matInvP = glm::inverse(glm::make_mat4x4(glMatP.data()));
  glm::mat4 matInvMVP(matInvMV * matInvP);

  glm::vec3 sunDir =
      glm::normalize(glm::vec3(glm::inverse(mWorldTransform) * glm::vec4(mSunDirection, 0)));

  // set uniforms ------------------------------------------------------------
  mAtmoShader.Bind();

  mAtmoShader.SetUniform(mUniforms.sunIlluminance, mSunIlluminance);
  mAtmoShader.SetUniform(mUniforms.sunDir, sunDir[0], sunDir[1], sunDir[2]);

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

  // if (mUseClouds && mCloudTexture) {
  //   mCloudTexture->Bind(GL_TEXTURE3);
  //   mAtmoShader.SetUniform(mUniforms.cloudTexture, 3);
  //   mAtmoShader.SetUniform(mUniforms.cloudAltitude, mCloudHeight);
  // }

  // if (mShadowMap) {
  //   int texUnitShadow = 8;
  //   mAtmoShader.SetUniform(
  //       mUniforms.shadowCascades, static_cast<int>(mShadowMap->getMaps().size()));
  //   for (size_t i = 0; i < mShadowMap->getMaps().size(); ++i) {
  //     mShadowMap->getMaps()[i]->Bind(
  //         static_cast<GLenum>(GL_TEXTURE0) + texUnitShadow + static_cast<int>(i));
  //     glUniform1i(mUniforms.shadowMaps.at(i), texUnitShadow + static_cast<int>(i));

  //     auto mat = mShadowMap->getShadowMatrices()[i];
  //     glUniformMatrix4fv(mUniforms.shadowProjectionMatrices.at(i), 1, GL_FALSE, mat.GetData());
  //   }
  // }

  glUniformMatrix4fv(mUniforms.inverseModelViewMatrix, 1, GL_FALSE, glm::value_ptr(matInvMV));
  glUniformMatrix4fv(
      mUniforms.inverseModelViewProjectionMatrix, 1, GL_FALSE, glm::value_ptr(matInvMVP));
  glUniformMatrix4fv(mUniforms.inverseProjectionMatrix, 1, GL_FALSE, glm::value_ptr(matInvP));
  // glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(matMV));
  // glUniformMatrix4fv(mUniforms.modelMatrix, 1, GL_FALSE, glm::value_ptr(matM));

  // Initialize eclipse shadow-related uniforms and textures.
  // mEclipseShadowReceiver->preRender();

  mModel->setUniforms(mAtmoShader.GetProgram(), 5);

  // draw --------------------------------------------------------------------
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // clean up ----------------------------------------------------------------

  // Reset eclipse shadow-related texture units.
  // mEclipseShadowReceiver->postRender();

  if (mHDRBuffer) {
    mHDRBuffer->getDepthAttachment()->Unbind(GL_TEXTURE0);
    mHDRBuffer->getCurrentReadAttachment()->Unbind(GL_TEXTURE1);
  } else {
    auto* viewport   = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
    auto const& data = mGBufferData[viewport];
    data.mDepthBuffer->Unbind(GL_TEXTURE0);
    data.mColorBuffer->Unbind(GL_TEXTURE1);
  }

  // if (mUseClouds && mCloudTexture) {
  //   mCloudTexture->Unbind(GL_TEXTURE3);
  // }

  mAtmoShader.Release();

  glDepthMask(GL_TRUE);

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Atmosphere::GetBoundingBox(VistaBoundingBox& bb) {
  float extend = mRadii[0];

  // Boundingbox is computed by translation an edge points
  std::array<float, 3> const fMin = {-extend, -extend, -extend};
  std::array<float, 3> const fMax = {extend, extend, extend};

  bb.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres
