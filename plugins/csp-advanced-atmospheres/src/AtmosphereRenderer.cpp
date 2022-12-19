////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Atmosphere.hpp"

#include "logger.hpp"

#include "../../../src/cs-core/EclipseShadowReceiver.hpp"
#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/Shadows.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "../../../src/cs-utils/filesystem.hpp"
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

namespace csp::advanced_atmospheres {

enum Luminance {
  // Render the spectral radiance at kLambdaR, kLambdaG, kLambdaB.
  NONE,
  // Render the sRGB luminance, using an approximate (on the fly) conversion
  // from 3 spectral radiance values only (see section 14.3 in <a href=
  // "https://arxiv.org/pdf/1612.04336.pdf">A Qualitative and Quantitative
  //  Evaluation of 8 Clear Sky Models</a>).
  APPROXIMATE,
  // Render the sRGB luminance, precomputed from 15 spectral radiance values
  // (see section 4.4 in <a href=
  // "http://www.oskee.wz.cz/stranka/uploads/SCCG10ElekKmoch.pdf">Real-time
  //  Spectral Scattering in Large-scale Natural Participating Media</a>).
  PRECOMPUTED
};

AtmosphereRenderer::AtmosphereRenderer(std::shared_ptr<Plugin::Settings> settings,
    std::shared_ptr<cs::core::EclipseShadowReceiver>                     eclipseShadowReceiver)
    : mPluginSettings(std::move(settings))
    , mEclipseShadowReceiver(std::move(eclipseShadowReceiver)) {

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

  constexpr double kPi                 = 3.1415926;
  constexpr double kSunAngularRadius   = 0.00935 / 2.0;
  constexpr double kLengthUnitInMeters = 1.0;

  constexpr bool      use_half_precision_    = false;
  constexpr bool      use_ozone_             = true;
  constexpr bool      use_combined_textures_ = false;
  constexpr Luminance use_luminance_         = Luminance::PRECOMPUTED;

  // Values from "Reference Solar Spectral Irradiance: ASTM G-173", ETR column
  // (see http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html),
  // summed and averaged in each bin (e.g. the value for 360nm is the average
  // of the ASTM G-173 values for all wavelengths between 360 and 370nm).
  // Values in W.m^-2.
  constexpr int    kLambdaMin           = 360;
  constexpr int    kLambdaMax           = 830;
  constexpr double kSolarIrradiance[48] = {1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054,
      1.6887, 1.61253, 1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
      1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533, 1.6965, 1.68194,
      1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482, 1.41018, 1.36775, 1.34188, 1.31429,
      1.28303, 1.26758, 1.2367, 1.2082, 1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992};
  // Values from http://www.iup.uni-bremen.de/gruppen/molspec/databases/
  // referencespectra/o3spectra2011/index.html for 233K, summed and averaged in
  // each bin (e.g. the value for 360nm is the average of the original values
  // for all wavelengths between 360 and 370nm). Values in m^2.
  constexpr double kOzoneCrossSection[48] = {1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27,
      2.763e-27, 5.52e-27, 8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26,
      9.016e-26, 1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
      4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25, 2.662e-25,
      2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26, 6.566e-26, 5.105e-26,
      4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26, 2.534e-26, 1.624e-26, 1.465e-26,
      2.078e-26, 1.383e-26, 7.105e-27};
  // From https://en.wikipedia.org/wiki/Dobson_unit, in molecules.m^-2.
  constexpr double kDobsonUnit = 2.687e20;
  // Maximum number density of ozone molecules, in m^-3 (computed so at to get
  // 300 Dobson units of ozone - for this we divide 300 DU by the integral of
  // the ozone density profile defined below, which is equal to 15km).
  constexpr double kMaxOzoneNumberDensity     = 300.0 * kDobsonUnit / 15000.0;
  constexpr double kBottomRadius              = 6371000.0;
  constexpr double kTopRadius                 = 6520000.0;
  constexpr double kRayleigh                  = 1.24062e-6;
  constexpr double kRayleighScaleHeight       = 8000.0;
  constexpr double kMieScaleHeight            = 1200.0;
  constexpr double kMieAngstromAlpha          = 0.0;
  constexpr double kMieAngstromBeta           = 5.328e-3;
  constexpr double kMieSingleScatteringAlbedo = 0.9;
  constexpr double kMiePhaseFunctionG         = 0.8;
  constexpr double kGroundAlbedo              = 0.1;
  const double     max_sun_zenith_angle       = (use_half_precision_ ? 102.0 : 120.0) / 180.0 * kPi;

  bruneton::DensityProfileLayer rayleigh_layer(0.0, 1.0, -1.0 / kRayleighScaleHeight, 0.0, 0.0);
  bruneton::DensityProfileLayer mie_layer(0.0, 1.0, -1.0 / kMieScaleHeight, 0.0, 0.0);
  // Density profile increasing linearly from 0 to 1 between 10 and 25km, and
  // decreasing linearly from 1 to 0 between 25 and 40km. This is an approximate
  // profile from http://www.kln.ac.lk/science/Chemistry/Teaching_Resources/
  // Documents/Introduction%20to%20atmospheric%20chemistry.pdf (page 10).
  std::vector<bruneton::DensityProfileLayer> ozone_density;
  ozone_density.push_back(
      bruneton::DensityProfileLayer(25000.0, 0.0, 0.0, 1.0 / 15000.0, -2.0 / 3.0));
  ozone_density.push_back(bruneton::DensityProfileLayer(0.0, 0.0, 0.0, -1.0 / 15000.0, 8.0 / 3.0));

  std::vector<double> wavelengths;
  std::vector<double> solar_irradiance;
  std::vector<double> rayleigh_scattering;
  std::vector<double> mie_scattering;
  std::vector<double> mie_extinction;
  std::vector<double> absorption_extinction;
  std::vector<double> ground_albedo;
  for (int l = kLambdaMin; l <= kLambdaMax; l += 10) {
    double lambda = static_cast<double>(l) * 1e-3; // micro-meters
    double mie    = kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);
    wavelengths.push_back(l);
    solar_irradiance.push_back(kSolarIrradiance[(l - kLambdaMin) / 10]);
    rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4));
    mie_scattering.push_back(mie * kMieSingleScatteringAlbedo);
    mie_extinction.push_back(mie);
    absorption_extinction.push_back(
        use_ozone_ ? kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) / 10] : 0.0);
    ground_albedo.push_back(kGroundAlbedo);
  }

  mModel.reset(new bruneton::Model(wavelengths, solar_irradiance, kSunAngularRadius, kBottomRadius,
      kTopRadius, {rayleigh_layer}, rayleigh_scattering, {mie_layer}, mie_scattering,
      mie_extinction, kMiePhaseFunctionG, ozone_density, absorption_extinction, ground_albedo,
      max_sun_zenith_angle, kLengthUnitInMeters, use_luminance_ == PRECOMPUTED ? 15 : 3,
      use_combined_textures_, use_half_precision_));
  glDisable(GL_CULL_FACE);
  mModel->Init();
  glEnable(GL_CULL_FACE);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AtmosphereRenderer::setSun(glm::vec3 const& direction, float illuminance) {
  mSunIlluminance = illuminance;
  mSunDirection   = direction;
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

void AtmosphereRenderer::updateShader() {
  mAtmoShader = VistaGLSLShader();

  auto sVert = cs::utils::filesystem::loadToString("../share/resources/shaders/atmosphere.vert");
  auto sFrag = cs::utils::filesystem::loadToString("../share/resources/shaders/atmosphere.frag");

  cs::utils::replaceString(sFrag, "USE_SHADOWMAP", std::to_string(mShadowMap != nullptr));
  cs::utils::replaceString(sFrag, "USE_CLOUDMAP", std::to_string(mUseClouds && mCloudTexture));
  cs::utils::replaceString(sFrag, "ENABLE_HDR", std::to_string(mHDRBuffer != nullptr));
  cs::utils::replaceString(sFrag, "HDR_SAMPLES",
      mHDRBuffer == nullptr ? "0" : std::to_string(mHDRBuffer->getMultiSamples()));

  // If the atmosphere should receive eclipse shadows, we need to inject the corresponding shader
  // source code snippet. If no eclipse shadow receiver was given, we just add a dummy method.
  if (mEclipseShadowReceiver) {
    cs::utils::replaceString(
        sFrag, "ECLIPSE_SHADER_SNIPPET", mEclipseShadowReceiver->getShaderSnippet());
  } else {
    cs::utils::replaceString(sFrag, "ECLIPSE_SHADER_SNIPPET",
        "vec3 getEclipseShadow(vec3 position) { return vec3(1); }");
  }

  mAtmoShader.InitVertexShaderFromString(sVert);
  mAtmoShader.InitFragmentShaderFromString(sFrag);

  glAttachShader(mAtmoShader.GetProgram(), mModel->shader());

  mAtmoShader.Link();

  mUniforms.sunDir         = mAtmoShader.GetUniformLocation("uSunDir");
  mUniforms.depthBuffer    = mAtmoShader.GetUniformLocation("uDepthBuffer");
  mUniforms.colorBuffer    = mAtmoShader.GetUniformLocation("uColorBuffer");
  mUniforms.cloudTexture   = mAtmoShader.GetUniformLocation("uCloudTexture");
  mUniforms.cloudAltitude  = mAtmoShader.GetUniformLocation("uCloudAltitude");
  mUniforms.shadowCascades = mAtmoShader.GetUniformLocation("uShadowCascades");
  mUniforms.sunIlluminance = mAtmoShader.GetUniformLocation("uSunIlluminance");

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
  mUniforms.modelMatrix                      = mAtmoShader.GetUniformLocation("uMatM");

  // We bind the eclipse shadow map to texture unit 4.
  if (mEclipseShadowReceiver) {
    mEclipseShadowReceiver->init(&mAtmoShader, 4);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AtmosphereRenderer::Do() {
  cs::utils::FrameTimings::ScopedTimer timer("Render Atmosphere");

  if (mShaderDirty || (mEclipseShadowReceiver && mEclipseShadowReceiver->needsRecompilation())) {
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

  if (mUseClouds && mCloudTexture) {
    mCloudTexture->Bind(GL_TEXTURE3);
    mAtmoShader.SetUniform(mUniforms.cloudTexture, 3);
    mAtmoShader.SetUniform(mUniforms.cloudAltitude, mCloudHeight);
  }

  if (mShadowMap) {
    int texUnitShadow = 8;
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
  glUniformMatrix4fv(mUniforms.modelMatrix, 1, GL_FALSE, glm::value_ptr(matM));

  // Initialize eclipse shadow-related uniforms and textures.
  if (mEclipseShadowReceiver) {
    mEclipseShadowReceiver->preRender();
  }

  mModel->SetProgramUniforms(mAtmoShader.GetProgram(), 5, 6, 7, 8);

  // draw --------------------------------------------------------------------
  mQuadVAO.Bind();
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  mQuadVAO.Release();

  // clean up ----------------------------------------------------------------

  // Reset eclipse shadow-related texture units.
  if (mEclipseShadowReceiver) {
    mEclipseShadowReceiver->postRender();
  }

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

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AtmosphereRenderer::GetBoundingBox(VistaBoundingBox& bb) {
  float extend = 6420000.F;

  // Boundingbox is computed by translation an edge points
  std::array<float, 3> const fMin = {-extend, -extend, -extend};
  std::array<float, 3> const fMax = {extend, extend, extend};

  bb.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::advanced_atmospheres
