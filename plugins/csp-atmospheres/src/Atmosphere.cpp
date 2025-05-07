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
#include "utils.hpp"

#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/Frustum.hpp"
#include "../../../src/cs-utils/filesystem.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaOGLExt/VistaTexture.h>

#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

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
    , mObjectName(std::move(objectName))
    , mEclipseShadowReceiver(
          std::make_shared<cs::core::EclipseShadowReceiver>(mAllSettings, mSolarSystem, false)) {

  // Recompile the shader if HDR mode was toggled.
  mEnableHDRConnection = mAllSettings->mGraphics.pEnableHDR.connectAndTouch([this](bool val) {
    if (val && !mHDRBuffer) {
      mHDRBuffer   = mGraphicsEngine->getHDRBuffer();
      mShaderDirty = true;
    } else if (!val && mHDRBuffer) {
      mHDRBuffer   = nullptr;
      mShaderDirty = true;
    }
  });

  // Attach this to the scene graph root.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mAtmosphereNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  mAtmosphereNode->SetIsEnabled(false);
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mAtmosphereNode.get(), static_cast<int>(cs::utils::DrawOrder::eAtmospheres));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Atmosphere::~Atmosphere() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mAtmosphereNode.get());

  mAllSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);

  if (mLimbLuminanceTexture != 0) {
    glDeleteTextures(1, &mLimbLuminanceTexture);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Atmosphere::configure(Plugin::Settings::Atmosphere const& settings) {
  auto object = mSolarSystem->getObject(mObjectName);
  if (object) {
    auto radii = object->getRadii();

    // Recreate the model if required.
    if (!mModel || settings.mModel != mSettings.mModel) {
      switch (settings.mModel.get()) {
      case Plugin::Settings::Atmosphere::Model::eCosmoScoutVR:
        mModel = std::make_unique<models::cosmoscout::Model>();
        break;
      case Plugin::Settings::Atmosphere::Model::eBruneton:
        mModel = std::make_unique<models::bruneton::Model>();
        break;
      }
    }

    // Reconfigure the model if any related parameter changed.
    if (mRadii != radii || settings.mTopAltitude != mSettings.mTopAltitude ||
        settings.mBottomAltitude != mSettings.mBottomAltitude ||
        settings.mModelSettings != mSettings.mModelSettings) {

      if (mModel->init(settings.mModelSettings, radii[0] + settings.mBottomAltitude.get(),
              radii[0] + settings.mTopAltitude)) {
        mShaderDirty = true;
      }
    }

    // Reload the cloud texture if required.
    if (mSettings.mCloudTexture != settings.mCloudTexture) {
      if (settings.mCloudTexture.has_value() && !settings.mCloudTexture.value().empty()) {
        mCloudTexture = cs::graphics::TextureLoader::loadFromFile(settings.mCloudTexture.value());
        mCloudTexture->Bind();
        glTexParameteri(mCloudTexture->GetTarget(), GL_TEXTURE_MAX_LOD, 5);
        mCloudTexture->Unbind();
      } else {
        mCloudTexture.reset();
      }
      mShaderDirty = true;
    }

    // Reload the limb luminance texture if required.
    if (mSettings.mLimbLuminanceTexture != settings.mLimbLuminanceTexture) {
      if (settings.mLimbLuminanceTexture.has_value() &&
          !settings.mLimbLuminanceTexture.value().empty()) {
        mLimbLuminanceTexture =
            std::get<0>(utils::read3DTexture(settings.mLimbLuminanceTexture.value()));
      } else {
        mLimbLuminanceTexture = 0;
      }
      mShaderDirty = true;
    }

    // Recreate the shader if required.
    if (mRadii != radii) {
      mRadii       = radii;
      mShaderDirty = true;
    }

    if (mSettings.mTopAltitude != settings.mTopAltitude ||
        mSettings.mBottomAltitude != settings.mBottomAltitude ||
        mSettings.mEnableWater != settings.mEnableWater ||
        mSettings.mEnableWaves != settings.mEnableWaves ||
        mSettings.mEnableClouds != settings.mEnableClouds ||
        mSettings.mEnableLimbLuminance != settings.mEnableLimbLuminance) {
      mShaderDirty = true;
    }

    mSettings = settings;

    if (mSettings.mRenderSkydome.get()) {
      renderSkyDome(mObjectName);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Atmosphere::createShader(ShaderType type, VistaGLSLShader& shader, Uniforms& uniforms) const {
  shader = VistaGLSLShader();

  auto sVert =
      cs::utils::filesystem::loadToString("../share/resources/shaders/csp-atmosphere.vert");
  auto sFrag =
      cs::utils::filesystem::loadToString("../share/resources/shaders/csp-atmosphere.frag");

  cs::utils::replaceString(sFrag, "SKYDOME_MODE", type == ShaderType::eSkyDome ? "1" : "0");
  cs::utils::replaceString(
      sFrag, "PLANET_RADIUS", std::to_string(mRadii[0] + mSettings.mBottomAltitude.get()));
  cs::utils::replaceString(
      sFrag, "ATMOSPHERE_RADIUS", std::to_string(mRadii[0] + mSettings.mTopAltitude));
  cs::utils::replaceString(
      sFrag, "ENABLE_CLOUDS", std::to_string(mSettings.mEnableClouds.get() && mCloudTexture));
  cs::utils::replaceString(sFrag, "ENABLE_LIMB_LUMINANCE",
      std::to_string(mSettings.mEnableLimbLuminance.get() && mLimbLuminanceTexture));
  cs::utils::replaceString(sFrag, "ENABLE_WATER", std::to_string(mSettings.mEnableWater.get()));
  cs::utils::replaceString(sFrag, "ENABLE_WAVES", std::to_string(mSettings.mEnableWaves.get()));
  cs::utils::replaceString(sFrag, "ENABLE_HDR", std::to_string(mHDRBuffer != nullptr));
  cs::utils::replaceString(sFrag, "HDR_SAMPLES",
      mHDRBuffer == nullptr ? "0" : std::to_string(mHDRBuffer->getMultiSamples()));
  cs::utils::replaceString(
      sFrag, "ECLIPSE_SHADER_SNIPPET", mEclipseShadowReceiver->getShaderSnippet());

  shader.InitVertexShaderFromString(sVert);
  shader.InitFragmentShaderFromString(sFrag);

  // Add the fragment shader from the atmospheric model.
  glAttachShader(shader.GetProgram(), mModel->getShader());

  shader.Link();

  uniforms.sunDir                    = shader.GetUniformLocation("uSunDir");
  uniforms.sunInfo                   = shader.GetUniformLocation("uSunInfo");
  uniforms.time                      = shader.GetUniformLocation("uTime");
  uniforms.depthBuffer               = shader.GetUniformLocation("uDepthBuffer");
  uniforms.colorBuffer               = shader.GetUniformLocation("uColorBuffer");
  uniforms.waterLevel                = shader.GetUniformLocation("uWaterLevel");
  uniforms.cloudTexture              = shader.GetUniformLocation("uCloudTexture");
  uniforms.cloudAltitude             = shader.GetUniformLocation("uCloudAltitude");
  uniforms.limbLuminanceTexture      = shader.GetUniformLocation("uLimbLuminanceTexture");
  uniforms.inverseModelViewMatrix    = shader.GetUniformLocation("uMatInvMV");
  uniforms.inverseProjectionMatrix   = shader.GetUniformLocation("uMatInvP");
  uniforms.scaleMatrix               = shader.GetUniformLocation("uMatScale");
  uniforms.modelMatrix               = shader.GetUniformLocation("uMatM");
  uniforms.modelViewProjectionMatrix = shader.GetUniformLocation("uMatMVP");
  uniforms.atmoPanoUniforms          = shader.GetUniformLocation("uAtmoPanoUniforms");
  uniforms.sunElevation              = shader.GetUniformLocation("sunElevation");
  uniforms.shadowCoordinates         = shader.GetUniformLocation("uShadowCoordinates");

  // We bind the eclipse shadow map to texture unit 3. The color and depth buffer are bound to 0 and
  // 1, 2 is used for the cloud map, 3 is used for the limb luminance texture.
  if (type == ShaderType::eAtmosphere) {
    mEclipseShadowReceiver->init(&shader, 4);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Atmosphere::updateShaders() {
  createShader(ShaderType::eAtmosphere, mAtmoShader, mAtmoUniforms);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Atmosphere::update(double time) {
  auto object = mSolarSystem->getObject(mObjectName);

  if (object && object->getIsBodyVisible() && mPluginSettings->mEnable.get()) {
    mTime           = time;
    mSunIlluminance = mSolarSystem->getSunIlluminance(object->getObserverRelativePosition());
    mSunLuminance   = mSolarSystem->getSunLuminance();
    mSunDirection   = mSolarSystem->getSunDirection(object->getObserverRelativePosition());
    mObserverRelativeTransformation = object->getObserverRelativeTransform();
    mSceneScale                     = mSolarSystem->getObserver().getScale();
    mEclipseShadowReceiver->update(*object);

    // This is a crude approximation of the overall scene brightness due to atmospheric scattering,
    // camera position and the Sun's position. It may be used for fake HDR effects such as dimming
    // stars.

    glm::dvec3 planet = object->getObserverRelativePosition() *
                        object->getRelativeScale(mSolarSystem->getObserver());
    double     dist     = glm::length(planet);
    glm::dvec3 toPlanet = planet / dist;

    // Altitude in [0.2x atmosphere boundary ... 5x atmosphere boundary] -> [0 ... 1]
    double heightInAtmosphere =
        std::min(1.0, std::max(0.0, (dist - object->getRadii()[0] - mSettings.mTopAltitude * 0.2) /
                                        (mSettings.mTopAltitude * 5.0)));

    // [noon ... midnight] -> [1 ... -1]
    double daySide = glm::dot(-toPlanet, glm::dvec3(mSunDirection));

    // Limit brightness when on night side (also in dusk and dawn time).
    daySide = std::pow(std::min(1.0, std::max(0.0, daySide + 1.0)), 50.0);

    // Reduce brightness in outer space.
    mGraphicsEngine->pApproximateSceneBrightness =
        static_cast<float>((1.0 - heightInAtmosphere) * daySide);

    mAtmosphereNode->SetIsEnabled(true);
  } else {
    mAtmosphereNode->SetIsEnabled(false);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Atmosphere::Do() {
  cs::utils::FrameStats::ScopedTimer          timer("Atmosphere of " + mObjectName);
  cs::utils::FrameStats::ScopedSamplesCounter samplesCounter("Atmosphere of " + mObjectName);

  if (mShaderDirty || mEclipseShadowReceiver->needsRecompilation()) {
    updateShaders();
    mShaderDirty = false;
  }

  // save current lighting and meterial state of the OpenGL state machine --------------------------
  glPushAttrib(GL_LIGHTING_BIT | GL_ENABLE_BIT);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glEnable(GL_TEXTURE_2D);
  glDepthMask(GL_FALSE);

  // get matrices and related values ---------------------------------------------------------------

  std::array<GLfloat, 16> glMatV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  // The atmosphere code works on spherical atmospheres. In order to support atmospheres around
  // ellipsoidal bodies, we apply a non-uniform scale to the atmosphere.
  glm::dmat4 matEllipsoid(
      1, 0, 0, 0, 0, mRadii[1] / mRadii[0], 0, 0, 0, 0, mRadii[2] / mRadii[0], 0, 0, 0, 0, 1);
  glm::dmat4 matInverseEllipsoid(
      1, 0, 0, 0, 0, mRadii[0] / mRadii[1], 0, 0, 0, 0, mRadii[0] / mRadii[2], 0, 0, 0, 0, 1);

  // We apply this non-uniform scaling to the observer-relative transformation of the planet.
  glm::dmat4 matM        = mObserverRelativeTransformation * matEllipsoid;
  glm::dmat4 matV        = glm::make_mat4x4(glMatV.data());
  glm::dmat4 matP        = glm::make_mat4x4(glMatP.data());
  glm::dmat4 matInvV     = glm::inverse(matV);
  glm::dmat4 matInvWorld = glm::inverse(mObserverRelativeTransformation);
  glm::dmat4 matInvMV    = matInverseEllipsoid * matInvWorld * matInvV;
  glm::mat4  matInvP     = glm::inverse(matP);
  glm::mat4  matMVP      = glm::mat4(matP * matV * matM);

  // Reconstructing the frame-buffer depth in the atmosphere shader is a bit involved as a simple
  // multiplication with matInvP would lead to coordinates in observer-relative coordinates which
  // are not non-uniformly scaled. To fix this, we have to create a matrix which applies this
  // non-uniform scaling to the reconstructed observer-relative coordinates.
  glm::dmat4 matScale = matV * mObserverRelativeTransformation * matInvMV;
  matScale            = glm::scale(matScale, glm::dvec3(mSceneScale));
  matScale[3]         = glm::vec4(0.0);

  glm::vec3 sunDir =
      glm::normalize(glm::vec3(matInverseEllipsoid * matInvWorld * glm::vec4(mSunDirection, 0)));

  // set uniforms ----------------------------------------------------------------------------------
  mAtmoShader.Bind();

  float sunRadius = float(mSolarSystem->getSun()->getRadii()[0]);
  float sunDist   = float(glm::length(mSolarSystem->pSunPosition.get()) * mSceneScale);
  float phiSun    = std::asin(sunRadius / sunDist);

  // If precomputed limb luminance is about to be used, we need to pass the eclipse-shadow map
  // coordinates to the shader as they are two of the three texture coordinates. We also need to
  // compute the approximate pixel width of the atmosphere ring as the precomputed limb luminance is
  // only used if the ring is pretty thin.
  if (mSettings.mEnableLimbLuminance.get() && mLimbLuminanceTexture) {
    glm::vec3 occDir = glm::vec3(matInvMV[3]);
    float     delta =
        std::acos(std::min(1.F, glm::dot(glm::normalize(occDir), glm::normalize(-sunDir))));
    float occDist      = glm::length(occDir);
    float planetRadius = float(mRadii[0] + mSettings.mBottomAltitude.get());
    float atmoRadius   = float(mRadii[0] + mSettings.mTopAltitude);
    float phiOcc       = std::asin(planetRadius / occDist);
    float phiAtmo      = std::asin(atmoRadius / occDist);
    float x            = 1.0F / (phiOcc / phiSun + 1.0F);
    float y            = 1.0F - delta / (phiOcc + phiSun);

    cs::utils::Frustum frustum;
    frustum.setFromMatrix(matP);

    std::array<GLint, 4> iViewport{};
    glGetIntegerv(GL_VIEWPORT, iViewport.data());

    float approxPixelSize = float(frustum.getHorizontalFOV() / iViewport[2]);
    float pixelWidth      = (phiAtmo - phiOcc) / approxPixelSize;
    if (occDist < atmoRadius) {
      x = -1.0;
    }

    mAtmoShader.SetUniform(mAtmoUniforms.shadowCoordinates, x, y, pixelWidth);
  }

  mAtmoShader.SetUniform(mAtmoUniforms.sunInfo, static_cast<float>(mSunLuminance),
      static_cast<float>(mSunIlluminance), phiSun);
  mAtmoShader.SetUniform(mAtmoUniforms.sunDir, sunDir[0], sunDir[1], sunDir[2]);

  // The noise shader does not like huge numbers. So we rather loop the time.
  mAtmoShader.SetUniform(mAtmoUniforms.time, static_cast<float>(std::fmod(mTime, 1.0e4)));

  if (mHDRBuffer) {
    mHDRBuffer->doPingPong();
    mHDRBuffer->bind();
  }

  auto depthBuffer = mGraphicsEngine->getCurrentDepthBufferAsTexture(false);
  auto colorBuffer = mGraphicsEngine->getCurrentColorBufferAsTexture(false);
  depthBuffer->Bind(GL_TEXTURE0);
  colorBuffer->Bind(GL_TEXTURE1);

  mAtmoShader.SetUniform(mAtmoUniforms.depthBuffer, 0);
  mAtmoShader.SetUniform(mAtmoUniforms.colorBuffer, 1);

  if (mSettings.mEnableWater.get()) {
    mAtmoShader.SetUniform(mAtmoUniforms.waterLevel,
        mSettings.mWaterLevel.get() * mAllSettings->mGraphics.pHeightScale.get() -
            static_cast<float>(mSettings.mBottomAltitude.get()));
  }

  if (mSettings.mEnableClouds.get() && mCloudTexture) {
    mCloudTexture->Bind(GL_TEXTURE2);
    mAtmoShader.SetUniform(mAtmoUniforms.cloudTexture, 2);
    mAtmoShader.SetUniform(mAtmoUniforms.cloudAltitude, mSettings.mCloudAltitude.get());
  }

  if (mSettings.mEnableLimbLuminance.get() && mLimbLuminanceTexture) {
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, mLimbLuminanceTexture);
    mAtmoShader.SetUniform(mAtmoUniforms.limbLuminanceTexture, 3);
  }

  glUniformMatrix4fv(
      mAtmoUniforms.inverseModelViewMatrix, 1, GL_FALSE, glm::value_ptr(glm::mat4(matInvMV)));
  glUniformMatrix4fv(mAtmoUniforms.scaleMatrix, 1, GL_FALSE, glm::value_ptr(glm::mat4(matScale)));
  glUniformMatrix4fv(mAtmoUniforms.inverseProjectionMatrix, 1, GL_FALSE, glm::value_ptr(matInvP));
  glUniformMatrix4fv(mAtmoUniforms.modelMatrix, 1, GL_FALSE, glm::value_ptr(glm::mat4(matM)));
  glUniformMatrix4fv(
      mAtmoUniforms.modelViewProjectionMatrix, 1, GL_FALSE, glm::value_ptr(glm::mat4(matMVP)));

  // Initialize eclipse shadow-related uniforms and textures.
  mEclipseShadowReceiver->preRender();

  // We bind the eclipse shadow map to texture unit 3. The color and depth buffer are bound to 0 and
  // 1, 2 is used for the cloud map, 3 is used for the limb luminance texture, then there are four
  // for the eclipse shadow receiver.
  mModel->setUniforms(mAtmoShader.GetProgram(), 8);

  // draw ------------------------------------------------------------------------------------------
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // clean up --------------------------------------------------------------------------------------

  // Reset eclipse shadow-related texture units.
  mEclipseShadowReceiver->postRender();

  depthBuffer->Unbind(GL_TEXTURE0);
  colorBuffer->Unbind(GL_TEXTURE1);

  if (mSettings.mEnableClouds.get() && mCloudTexture) {
    mCloudTexture->Unbind(GL_TEXTURE2);
  }

  if (mSettings.mEnableClouds.get() && mLimbLuminanceTexture) {
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, 0);
  }

  mAtmoShader.Release();

  glDepthMask(GL_TRUE);

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Atmosphere::GetBoundingBox(VistaBoundingBox& bb) {
  float extend = static_cast<float>(std::max(std::max(mRadii[0], mRadii[1]), mRadii[2]));

  // Boundingbox is computed by translation an edge points
  std::array<float, 3> const fMin = {-extend, -extend, -extend};
  std::array<float, 3> const fMax = {extend, extend, extend};

  bb.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Atmosphere::renderSkyDome(std::string const& name) const {
  const int SIZE = 512;

  VistaGLSLShader shader;
  Uniforms        uniforms;
  createShader(ShaderType::eSkyDome, shader, uniforms);

  GLuint texture;
  glGenTextures(1, &texture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SIZE, SIZE, 0, GL_RGBA, GL_FLOAT, nullptr);

  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);

  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glViewport(0, 0, SIZE, SIZE);
  glScissor(0, 0, SIZE, SIZE);

  glPushAttrib(GL_LIGHTING_BIT | GL_ENABLE_BIT);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glEnable(GL_TEXTURE_2D);
  glDepthMask(GL_FALSE);

  shader.Bind();

  double const sunLuminousPower = 3.75e28;
  double       sunDist          = 149597870700;
  double       sunIlluminance   = sunLuminousPower / (sunDist * sunDist * 4.0 * glm::pi<double>());

  shader.SetUniform(uniforms.sunInfo, 0.f, static_cast<float>(sunIlluminance), 0.f);

  std::vector<float> pixels(SIZE * SIZE * 4);

  std::vector<float> elevation = {0.0, 45.0, 75.0, 90.0};

  for (float e : elevation) {
    shader.SetUniform(uniforms.sunElevation, e);
    mModel->setUniforms(shader.GetProgram(), 4);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glReadPixels(0, 0, SIZE, SIZE, GL_RGBA, GL_FLOAT, &pixels[0]);

    stbi_write_hdr(std::format("{}_{}.hdr", name, e).c_str(), SIZE, SIZE, 4, pixels.data());
  }

  glDepthMask(GL_TRUE);

  shader.Release();

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(1, &texture);

  glPopAttrib();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres
