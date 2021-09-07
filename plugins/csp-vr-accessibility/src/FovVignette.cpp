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

  // create shaders & get uniform locations
  mShaderFade.InitVertexShaderFromString(VERT_SHADER);
  mShaderFade.InitFragmentShaderFromString(FRAG_SHADER_FADE);
  mShaderFade.Link();
  mUniforms.fade.aspect      = mShaderFade.GetUniformLocation("uAspect");
  mUniforms.fade.fade        = mShaderFade.GetUniformLocation("uFade");
  mUniforms.fade.color       = mShaderFade.GetUniformLocation("uCustomColor");
  mUniforms.fade.innerRadius = mShaderFade.GetUniformLocation("uInnerRadius");
  mUniforms.fade.outerRadius = mShaderFade.GetUniformLocation("uOuterRadius");
  mUniforms.fade.debug       = mShaderFade.GetUniformLocation("uDebug");

  mShaderDynRad.InitVertexShaderFromString(VERT_SHADER);
  mShaderDynRad.InitFragmentShaderFromString(FRAG_SHADER_DYNRAD);
  mShaderDynRad.Link();
  mUniforms.dynamic.aspect       = mShaderDynRad.GetUniformLocation("uAspect");
  mUniforms.dynamic.normVelocity = mShaderDynRad.GetUniformLocation("uNormVelocity");
  mUniforms.dynamic.color        = mShaderDynRad.GetUniformLocation("uCustomColor");
  mUniforms.dynamic.innerRadius  = mShaderDynRad.GetUniformLocation("uInnerRadius");
  mUniforms.dynamic.outerRadius  = mShaderDynRad.GetUniformLocation("uOuterRadius");
  mUniforms.dynamic.debug        = mShaderDynRad.GetUniformLocation("uDebug");

  mShaderFadeVertOnly.InitVertexShaderFromString(VERT_SHADER);
  mShaderFadeVertOnly.InitFragmentShaderFromString(FRAG_SHADER_FADE_VERTONLY);
  mShaderFadeVertOnly.Link();
  mUniforms.fadeVertical.aspect      = mShaderFadeVertOnly.GetUniformLocation("uAspect");
  mUniforms.fadeVertical.fade        = mShaderFadeVertOnly.GetUniformLocation("uFade");
  mUniforms.fadeVertical.color       = mShaderFadeVertOnly.GetUniformLocation("uCustomColor");
  mUniforms.fadeVertical.innerRadius = mShaderFadeVertOnly.GetUniformLocation("uInnerRadius");
  mUniforms.fadeVertical.outerRadius = mShaderFadeVertOnly.GetUniformLocation("uOuterRadius");
  mUniforms.fadeVertical.debug       = mShaderFadeVertOnly.GetUniformLocation("uDebug");

  mShaderDynRadVertOnly.InitVertexShaderFromString(VERT_SHADER);
  mShaderDynRadVertOnly.InitFragmentShaderFromString(FRAG_SHADER_DYNRAD_VERTONLY);
  mShaderDynRadVertOnly.Link();
  mUniforms.dynamicVertical.aspect      = mShaderDynRadVertOnly.GetUniformLocation("uAspect");
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
  mFadeAnimation.mDirection = cs::utils::AnimationDirection::eLinear;
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

  std::array<GLint, 4> viewport{};
  glGetIntegerv(GL_VIEWPORT, viewport.data());
  float aspect =static_cast<float>(viewport.at(3)) / static_cast<float>(viewport.at(2));

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
    shader.SetUniform(uniformLocs.aspect, aspect);
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
    shader.SetUniform(uniformLocs.aspect, aspect);
    shader.SetUniform(uniformLocs.fade, mFadeAnimation.get(getNow()));
    glUniform4fv(uniformLocs.color, 1,
        glm::value_ptr(Plugin::GetColorFromHexString(mVignetteSettings.mColor.get())));
    shader.SetUniform(uniformLocs.innerRadius, mVignetteSettings.mInnerRadius.get());
    shader.SetUniform(uniformLocs.outerRadius, mVignetteSettings.mOuterRadius.get());
    shader.SetUniform(uniformLocs.debug, mVignetteSettings.mDebug.get());
  }

  // draw
  glPushAttrib(GL_ENABLE_BIT | GL_BLEND | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDisable(GL_DEPTH_TEST);
  glDepthMask(false);

  mVAO.Bind();
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  mVAO.Release();

  glPopAttrib();

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

  double currentTime = getNow();

  // check for movement changes
  if (mIsMoving && velocity < mVignetteSettings.mLowerVelocityThreshold.get()) {
    // observer started moving
    mIsMoving = false;
    mLastChange = currentTime;
  } else if (!mIsMoving && velocity > mVignetteSettings.mLowerVelocityThreshold.get()) {
    // observer stopped moving
    mIsMoving = true;
    mLastChange = currentTime;
  }

  // check if deadzone has passed and tracker indicates animation needed
  if (currentTime - mVignetteSettings.mFadeDeadzone.get() >= mLastChange) {
    mFadeAnimation.mStartValue = mFadeAnimation.get(currentTime);
    mFadeAnimation.mStartTime  = currentTime;
    mFadeAnimation.mEndTime    = currentTime + mVignetteSettings.mFadeDuration.get();

    if (mIsMoving) {
      mFadeAnimation.mEndValue   = 1.0F;
    } else {
      mFadeAnimation.mEndValue   = 0.0F;
    }

     mLastChange = std::numeric_limits<double>::max();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double FovVignette::getNow() {
  boost::posix_time::ptime const EPOCH(boost::gregorian::date(1970,1,1));
  auto delta = boost::posix_time::microsec_clock::universal_time() - EPOCH;
  return delta.total_microseconds() / 1000000.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::vraccessibility
