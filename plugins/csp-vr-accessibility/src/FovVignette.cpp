////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FovVignette.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GraphicsEngine.hpp"
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaDisplaySystem.h>
#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/InteractionManager/VistaUserPlatform.h>
#include <VistaKernel/VistaFrameLoop.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <glm/gtc/type_ptr.hpp>

#include <utility>

namespace csp::vraccessibility {

FovVignette::FovVignette(std::shared_ptr<cs::core::SolarSystem> solarSystem,
    Plugin::Settings::Vignette&                                 vignetteSettings)
    : mSolarSystem(std::move(solarSystem))
    , mVignetteSettings(vignetteSettings) {

  // create quad
  std::array<float, 8> const data{-1, 1, 1, 1, -1, -1, 1, -1};

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);
  mVBO.Release();

  // positions
  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0, &mVBO);

  // create textures
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

  // create shaders & get uniform locations
  mShaderFade.InitVertexShaderFromString(VERT_SHADER);
  mShaderFade.InitFragmentShaderFromString(FRAG_SHADER_FADE);
  mShaderFade.Link();
  mUniforms.fade.texture     = mShaderFade.GetUniformLocation("uTexture");
  mUniforms.fade.fade        = mShaderFade.GetUniformLocation("uFade");
  mUniforms.fade.color       = mShaderFade.GetUniformLocation("uCustomColor");
  mUniforms.fade.innerRadius = mShaderFade.GetUniformLocation("uInnerRadius");
  mUniforms.fade.outerRadius = mShaderFade.GetUniformLocation("uOuterRadius");
  mUniforms.fade.debug       = mShaderFade.GetUniformLocation("uDebug");

  mShaderDynRad.InitVertexShaderFromString(VERT_SHADER);
  mShaderDynRad.InitFragmentShaderFromString(FRAG_SHADER_DYNRAD);
  mShaderDynRad.Link();
  mUniforms.dynamic.texture      = mShaderDynRad.GetUniformLocation("uTexture");
  mUniforms.dynamic.normVelocity = mShaderDynRad.GetUniformLocation("uNormVelocity");
  mUniforms.dynamic.color        = mShaderDynRad.GetUniformLocation("uCustomColor");
  mUniforms.dynamic.innerRadius  = mShaderDynRad.GetUniformLocation("uInnerRadius");
  mUniforms.dynamic.outerRadius  = mShaderDynRad.GetUniformLocation("uOuterRadius");
  mUniforms.dynamic.debug        = mShaderDynRad.GetUniformLocation("uDebug");

  mShaderFadeVertOnly.InitVertexShaderFromString(VERT_SHADER);
  mShaderFadeVertOnly.InitFragmentShaderFromString(FRAG_SHADER_FADE_VERTONLY);
  mShaderFadeVertOnly.Link();
  mUniforms.fadeVertical.texture     = mShaderFadeVertOnly.GetUniformLocation("uTexture");
  mUniforms.fadeVertical.fade        = mShaderFadeVertOnly.GetUniformLocation("uFade");
  mUniforms.fadeVertical.color       = mShaderFadeVertOnly.GetUniformLocation("uCustomColor");
  mUniforms.fadeVertical.innerRadius = mShaderFadeVertOnly.GetUniformLocation("uInnerRadius");
  mUniforms.fadeVertical.outerRadius = mShaderFadeVertOnly.GetUniformLocation("uOuterRadius");
  mUniforms.fadeVertical.debug       = mShaderFadeVertOnly.GetUniformLocation("uDebug");

  mShaderDynRadVertOnly.InitVertexShaderFromString(VERT_SHADER);
  mShaderDynRadVertOnly.InitFragmentShaderFromString(FRAG_SHADER_DYNRAD_VERTONLY);
  mShaderDynRadVertOnly.Link();
  mUniforms.dynamicVertical.texture = mShaderDynRadVertOnly.GetUniformLocation("uTexture");
  mUniforms.dynamicVertical.normVelocity =
      mShaderDynRadVertOnly.GetUniformLocation("uNormVelocity");
  mUniforms.dynamicVertical.color       = mShaderDynRadVertOnly.GetUniformLocation("uCustomColor");
  mUniforms.dynamicVertical.innerRadius = mShaderDynRadVertOnly.GetUniformLocation("uInnerRadius");
  mUniforms.dynamicVertical.outerRadius = mShaderDynRadVertOnly.GetUniformLocation("uOuterRadius");
  mUniforms.dynamicVertical.debug       = mShaderDynRadVertOnly.GetUniformLocation("uDebug");

  // add to scenegraph
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  auto* platform = GetVistaSystem()
                       ->GetPlatformFor(GetVistaSystem()->GetDisplayManager()->GetDisplaySystem())
                       ->GetPlatformNode();
  mGLNode.reset(pSG->NewOpenGLNode(platform, this));

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eGui) - 1);

  // init animation housekeeping
  mFadeAnimation    = cs::utils::AnimatedValue(0.0F, 0.0F, 0.0, 0.0);
  mLastChange       = 0.0;
  mAnimationTracker = 0;
  mIsStill          = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FovVignette::~FovVignette() {
  auto* platform = GetVistaSystem()
                       ->GetPlatformFor(GetVistaSystem()->GetDisplayManager()->GetDisplaySystem())
                       ->GetPlatformNode();
  platform->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FovVignette::configure(Plugin::Settings::Vignette& vignetteSettings) {
  mVignetteSettings = vignetteSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FovVignette::Do() {
  // do nothing if vignette is disabled
  if (!mVignetteSettings.mEnabled.get()) {
    return true;
  }

  cs::utils::FrameTimings::ScopedTimer timer("VRAccessibility-FovVignette");

  // copy depth buffer
  std::array<GLint, 4> iViewport{};
  glGetIntegerv(GL_VIEWPORT, iViewport.data());

  auto*       viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  auto const& data     = mGBufferData[viewport];

  data.mDepthBuffer->Bind();
  glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, iViewport.at(0), iViewport.at(1),
      iViewport.at(2), iViewport.at(3), 0);
  data.mColorBuffer->Bind();
  glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, iViewport.at(0), iViewport.at(1), iViewport.at(2),
      iViewport.at(3), 0);

  // set uniforms
  // check if dynamical vignette
  if (mVignetteSettings.mUseDynamicRadius.get()) {
    // check if vertical only (or circular vignetting), select shader accordingly
    VistaGLSLShader& shader =
        mVignetteSettings.mUseVerticalOnly.get() ? mShaderDynRadVertOnly : mShaderDynRad;
    auto uniformLocs =
        mVignetteSettings.mUseVerticalOnly.get() ? mUniforms.dynamicVertical : mUniforms.dynamic;

    shader.Bind();

    // set uniforms for dynamical vignette
    shader.SetUniform(uniformLocs.texture, 0);
    shader.SetUniform(uniformLocs.normVelocity, mNormalizedVelocity);
    glUniform4fv(uniformLocs.color, 1,
        glm::value_ptr(Plugin::GetColorFromHexString(mVignetteSettings.mColor.get())));
    shader.SetUniform(uniformLocs.innerRadius,
        // override current radius if debug enabled
        mVignetteSettings.mDebug.get() ? mVignetteSettings.mInnerRadius.get()
                                       : mCurrentInnerRadius);
    shader.SetUniform(uniformLocs.outerRadius,
        // override current radius if debug enabled
        mVignetteSettings.mDebug.get() ? mVignetteSettings.mOuterRadius.get()
                                       : mCurrentOuterRadius);
    shader.SetUniform(uniformLocs.debug, mVignetteSettings.mDebug.get());
  } else {
    // check if vertical only (or circular vignetting), select shader accordingly
    VistaGLSLShader& shader =
        mVignetteSettings.mUseVerticalOnly.get() ? mShaderFadeVertOnly : mShaderFade;
    auto uniformLocs =
        mVignetteSettings.mUseVerticalOnly.get() ? mUniforms.fadeVertical : mUniforms.fade;

    shader.Bind();

    // set uniforms for static vignette
    shader.SetUniform(uniformLocs.texture, 0);
    double currentTime =
        cs::utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time());
    shader.SetUniform(uniformLocs.fade, mFadeAnimation.get(currentTime));
    glUniform4fv(uniformLocs.color, 1,
        glm::value_ptr(Plugin::GetColorFromHexString(mVignetteSettings.mColor.get())));
    shader.SetUniform(uniformLocs.innerRadius, mVignetteSettings.mInnerRadius.get());
    shader.SetUniform(uniformLocs.outerRadius, mVignetteSettings.mOuterRadius.get());
    shader.SetUniform(uniformLocs.debug, mVignetteSettings.mDebug.get());
  }

  // bind texture
  data.mColorBuffer->Bind(GL_TEXTURE0);

  // draw
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDisable(GL_DEPTH_TEST);

  mVAO.Bind();
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  mVAO.Release();

  // clean up
  data.mDepthBuffer->Unbind(GL_TEXTURE0);
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);

  // release shader
  glUseProgram(0);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FovVignette::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float FovVignette::getNewRadius(
    float innerOuterRadius, float normVelocity, float lastRadius, double dT) {

  // targetRadius based on interpolation between maxRadius and innerOuterRadius
  float targetRadius = ((1 - normVelocity) * std::sqrt(2.0f)) + (normVelocity * innerOuterRadius);

  // newRadius increased towards targetRadius but limited to ~targetDiff changes
  return (0.99F * lastRadius) + (0.01F * targetRadius);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FovVignette::updateDynamicRadiusVignette() {
  // get simulation variables
  float velocity = mSolarSystem->pCurrentObserverSpeed.get() /
                   static_cast<float>(mSolarSystem->getObserver().getAnchorScale());
  auto now = std::chrono::high_resolution_clock::now();

  mNormalizedVelocity = (velocity - mVignetteSettings.mLowerVelocityThreshold.get()) /
                        (mVignetteSettings.mUpperVelocityThreshold.get() -
                            mVignetteSettings.mLowerVelocityThreshold.get());
  auto deltaTime = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - mLastTime).count());
  mLastTime = now;

  // clamp NormalizedVelocity to upper threshold
  mNormalizedVelocity = (mNormalizedVelocity > 1.0F) ? 1.0F : mNormalizedVelocity;

  mCurrentInnerRadius = getNewRadius(
      mVignetteSettings.mInnerRadius.get(), mNormalizedVelocity, mLastInnerRadius, deltaTime);
  mCurrentOuterRadius = getNewRadius(
      mVignetteSettings.mOuterRadius.get(), mNormalizedVelocity, mLastOuterRadius, deltaTime);

  // update variables
  mLastInnerRadius = mCurrentInnerRadius;
  mLastOuterRadius = mCurrentOuterRadius;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FovVignette::updateFadeAnimatedVignette() {
  // get simulation variables
  float velocity = mSolarSystem->pCurrentObserverSpeed.get() /
                   static_cast<float>(mSolarSystem->getObserver().getAnchorScale());
  double currentTime =
      cs::utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time());

  // check for movement changes
  if (mIsStill && velocity > mVignetteSettings.mLowerVelocityThreshold.get()) {
    // observer started moving
    mAnimationTracker += 1;
    mLastChange = currentTime;
  } else if (!mIsStill && velocity < mVignetteSettings.mLowerVelocityThreshold.get()) {
    // observer stopped moving
    mAnimationTracker -= 1;
    mLastChange = currentTime;
  }

  // update mIsStill
  mIsStill = (velocity < mVignetteSettings.mLowerVelocityThreshold.get());

  // check if deadzone has passed and tracker indicates animation needed
  if (mAnimationTracker != 0 && currentTime > mLastChange + mVignetteSettings.mFadeDeadzone.get()) {
    if (mAnimationTracker > 0) {
      // observer started moving
      mFadeAnimation.mStartValue = 0.0F;
      mFadeAnimation.mEndValue   = 1.0F;
      mFadeAnimation.mStartTime  = currentTime;
      mFadeAnimation.mEndTime    = currentTime + mVignetteSettings.mFadeDuration.get();
      // reset tracker
      mAnimationTracker = 0;
    } else {
      // observer stopped moving
      mFadeAnimation.mStartValue = 1.0F;
      mFadeAnimation.mEndValue   = 0.0F;
      mFadeAnimation.mStartTime  = currentTime;
      mFadeAnimation.mEndTime    = currentTime + mVignetteSettings.mFadeDuration.get();
      // reset tracker
      mAnimationTracker = 0;
    }
  }
}

} // namespace csp::vraccessibility
