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

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaDisplaySystem.h>
#include <VistaKernel/InteractionManager/VistaUserPlatform.h>
#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaFrameLoop.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <glm/gtc/type_ptr.hpp>

#include <utility>

namespace csp::vraccessibility {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* FovVignette::VERT_SHADER = R"(
#version 330

// inputs
layout(location = 0) in vec2 iQuadPos;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;

void main()
{
    vTexCoords  = vec2( (iQuadPos.x + 1) / 2,
                       (iQuadPos.y + 1) / 2 );
    vPosition   = vec3(iQuadPos.x, iQuadPos.y, -0.01);
    gl_Position = vec4(vPosition, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* FovVignette::FRAG_SHADER_FADE = R"(
#version 330

uniform sampler2D uTexture;
uniform float uFade;
uniform vec4 uCustomColor;
uniform float uInnerRadius;
uniform float uOuterRadius;
uniform bool uDebug;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

void main()
{
    if (uFade == 0 && !uDebug ) { discard; }

    vec2 texSize = textureSize(uTexture, 0);
    float ratio = texSize.y / texSize.x;

    float dist = sqrt(vPosition.x * vPosition.x + ratio * ratio * vPosition.y * vPosition.y);
    if (dist < uInnerRadius ) { discard; }
    float r = ((dist - uInnerRadius) / (uOuterRadius - uInnerRadius));
    oColor = mix(texture(uTexture, vTexCoords), uCustomColor, r);
    if (dist > uOuterRadius) {
      oColor.rgb = uCustomColor.rgb;
    }

    if ( !uDebug ) { oColor.a = uFade; }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* FovVignette::FRAG_SHADER_DYNRAD = R"(
#version 330

uniform sampler2D uTexture;
uniform float uNormVelocity;
uniform vec4 uCustomColor;
uniform float uInnerRadius;
uniform float uOuterRadius;
uniform bool uDebug;
float radiusInner = uInnerRadius;
float radiusOuter = uOuterRadius;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

void main()
{
    if (uNormVelocity <= 0 && !uDebug ) { discard; }

    vec2 texSize = textureSize(uTexture, 0);
    float ratio = texSize.y / texSize.x;

    float dist = sqrt(vPosition.x * vPosition.x + ratio * ratio * vPosition.y * vPosition.y);

    if (dist < uInnerRadius ) { discard; }
    float r = ((dist - uInnerRadius) / (uOuterRadius - uInnerRadius));
    oColor = mix(texture(uTexture, vTexCoords), uCustomColor, r);
    if (dist > uOuterRadius) {
      oColor.rgb = uCustomColor.rgb;
    }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

FovVignette::FovVignette(std::shared_ptr<cs::core::SolarSystem> solarSystem)
    : mSolarSystem(std::move(solarSystem)) {

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

  // create shaders
  mShaderFade.InitVertexShaderFromString(VERT_SHADER);
  mShaderFade.InitFragmentShaderFromString(FRAG_SHADER_FADE);
  mShaderFade.Link();

  mShaderDynRad.InitVertexShaderFromString(VERT_SHADER);
  mShaderDynRad.InitFragmentShaderFromString(FRAG_SHADER_DYNRAD);
  mShaderDynRad.Link();

  // add to scenegraph
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  auto* platform = GetVistaSystem()
                       ->GetPlatformFor(GetVistaSystem()->GetDisplayManager()->GetDisplaySystem())
                       ->GetPlatformNode();
  mGLNode.reset(pSG->NewOpenGLNode(platform, this));
  
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eGui) - 1
      );

  // init animation housekeeping
  mFadeAnimation = cs::utils::AnimatedValue( 0.0F, 0.0F, 0.0, 0.0 );
  mLastChange = 0.0;
  mAnimationTracker = 0;
  mIsStill = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FovVignette::~FovVignette() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FovVignette::configure(std::shared_ptr<Plugin::Settings> settings) {
  mVignetteSettings = settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FovVignette::Do() {
  if (!mVignetteSettings->mFovVignetteEnabled.get()){
    return true;
  }

  cs::utils::FrameTimings::ScopedTimer timer("VRAccessibility-FovVignette");

  // copy depth buffer
  std::array<GLint, 4> iViewport{};
  glGetIntegerv(GL_VIEWPORT, iViewport.data());

  auto* viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  auto const& data = mGBufferData[viewport];

  data.mDepthBuffer->Bind();
  glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, iViewport.at(0), iViewport.at(1),
                   iViewport.at(2), iViewport.at(3), 0);
  data.mColorBuffer->Bind();
  glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, iViewport.at(0), iViewport.at(1), iViewport.at(2),
                   iViewport.at(3), 0);

  //////////////////////////////////////////////////////////////////////////////////////////////////

  // set uniforms
  if (mVignetteSettings->mFovVignetteUseDynamicRadius.get()) {
    mShaderDynRad.Bind();

    mShaderDynRad.SetUniform(
        mShaderDynRad.GetUniformLocation("uTexture"), 0
    );
    mShaderDynRad.SetUniform(
        mShaderDynRad.GetUniformLocation("uNormVelocity"), mNormalizedVelocity
    );
    glUniform4fv(
        mShaderDynRad.GetUniformLocation("uCustomColor"), 1, glm::value_ptr(Plugin::GetColorFromHexString(mVignetteSettings->mFovVignetteColor.get()))
    );
    mShaderDynRad.SetUniform(
        mShaderDynRad.GetUniformLocation("uInnerRadius"), mCurrentInnerRadius
    );
    mShaderDynRad.SetUniform(
        mShaderDynRad.GetUniformLocation("uOuterRadius"), mCurrentOuterRadius
    );
    mShaderDynRad.SetUniform(
        mShaderDynRad.GetUniformLocation("uDebug"), mVignetteSettings->mFovVignetteDebug.get()
    );
  }
  else {
    mShaderFade.Bind();

    mShaderFade.SetUniform(
        mShaderFade.GetUniformLocation("uTexture"), 0
        );
    double currentTime = cs::utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time());
    mShaderFade.SetUniform(
        mShaderFade.GetUniformLocation("uFade"), mFadeAnimation.get(currentTime)
        );
    glUniform4fv(
        mShaderFade.GetUniformLocation("uCustomColor"), 1, glm::value_ptr(Plugin::GetColorFromHexString(mVignetteSettings->mFovVignetteColor.get()))
        );
    mShaderFade.SetUniform(
        mShaderFade.GetUniformLocation("uInnerRadius"), mVignetteSettings->mFovVignetteInnerRadius.get()
        );
    mShaderFade.SetUniform(
        mShaderFade.GetUniformLocation("uOuterRadius"), mVignetteSettings->mFovVignetteOuterRadius.get()
        );
    mShaderFade.SetUniform(
        mShaderFade.GetUniformLocation("uDebug"), mVignetteSettings->mFovVignetteDebug.get()
        );
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

  if(mVignetteSettings->mFovVignetteUseDynamicRadius.get()) {
    mShaderDynRad.Release();
  }
  else {
    mShaderFade.Release();
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FovVignette::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float FovVignette::getNewRadius(float innerOuterRadius, float normVelocity, float lastRadius, double dT) {
  // target change from last radius (in %) per target time delta (avg frame time)
  float targetDiff = 0.1F;
  auto targetDtime = static_cast<float>(GetVistaSystem()->GetFrameLoop()->GetAverageLoopTime());
  float diffPerTargetDt = targetDiff / targetDtime;

  // (dT / targetDtime) >1 if dT > targetDtime, <1 if dT < targetDtime
  float x = diffPerTargetDt * (static_cast<float>(dT) / targetDtime);

  // targetRadius based on interpolation between maxRadius and innerOuterRadius
  float targetRadius = ((1 - normVelocity) * sqrtf(2)) + (normVelocity * innerOuterRadius);

  // newRadius increased towards targetRadius but limited to ~targetDiff changes
  return (0.99F * lastRadius) + (0.01F * targetRadius);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FovVignette::updateDynamicRadiusVignette() {
  // get simulation variables
  float velocity = mSolarSystem->pCurrentObserverSpeed.get() / static_cast<float>(mSolarSystem->getObserver().getAnchorScale());
  auto now = std::chrono::high_resolution_clock::now();

  mNormalizedVelocity = (velocity - mVignetteSettings->mFovVignetteLowerVelocityThreshold.get()) / (mVignetteSettings->mFovVignetteUpperVelocityThreshold.get() - mVignetteSettings->mFovVignetteLowerVelocityThreshold.get());
  auto deltaTime = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - mLastTime).count());
  mLastTime = now;

  // clamp NormalizedVelocity to upper threshold
  mNormalizedVelocity = (mNormalizedVelocity > 1.0F)? 1.0F : mNormalizedVelocity;

  mCurrentInnerRadius = getNewRadius(mVignetteSettings->mFovVignetteInnerRadius.get(), mNormalizedVelocity, mLastInnerRadius, deltaTime);
  mCurrentOuterRadius = getNewRadius(mVignetteSettings->mFovVignetteOuterRadius.get(), mNormalizedVelocity, mLastOuterRadius, deltaTime);
  
  mLastInnerRadius = mCurrentInnerRadius;
  mLastOuterRadius = mCurrentOuterRadius;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FovVignette::updateFadeAnimatedVignette() {
  // get simulation variables
  float velocity = mSolarSystem->pCurrentObserverSpeed.get() / static_cast<float>(mSolarSystem->getObserver().getAnchorScale());
  double currentTime = cs::utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time());

  // check for movement changes
  if ( mIsStill && velocity > mVignetteSettings->mFovVignetteLowerVelocityThreshold.get() ) {
    // observer started moving
    mAnimationTracker += 1;
    mLastChange = currentTime;
  }
  else if ( !mIsStill && velocity < mVignetteSettings->mFovVignetteLowerVelocityThreshold.get() ) {
    // observer stopped moving
    mAnimationTracker -= 1;
    mLastChange = currentTime;
  }

  // update mIsStill
  mIsStill = (velocity < mVignetteSettings->mFovVignetteLowerVelocityThreshold.get());

  // check if deadzone has passed and tracker indicates animation needed
  if ( mAnimationTracker != 0 && currentTime > mLastChange + mVignetteSettings->mFovVignetteFadeDeadzone.get()) {
    if ( mAnimationTracker > 0 ) {
      // observer started moving
      mFadeAnimation.mStartValue  = 0.0F;
      mFadeAnimation.mEndValue    = 1.0F;
      mFadeAnimation.mStartTime   = currentTime;
      mFadeAnimation.mEndTime     = currentTime + mVignetteSettings->mFovVignetteFadeDuration.get();
      // reset tracker
      mAnimationTracker = 0;
    }
    else {
      // observer stopped moving
      mFadeAnimation.mStartValue  = 1.0F;
      mFadeAnimation.mEndValue    = 0.0F;
      mFadeAnimation.mStartTime   = currentTime;
      mFadeAnimation.mEndTime     = currentTime + mVignetteSettings->mFovVignetteFadeDuration.get();
      // reset tracker
      mAnimationTracker = 0;
    }
  }
}

} // namespace csp::vraccessibility