////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "EclipseShadowReceiver.hpp"

#include "../cs-graphics/EclipseShadowMap.hpp"
#include "../cs-scene/CelestialObject.hpp"
#include "../cs-utils/filesystem.hpp"
#include "../cs-utils/utils.hpp"
#include "GraphicsEngine.hpp"
#include "SolarSystem.hpp"

#include <VistaOGLExt/VistaGLSLShader.h>
#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace cs::core {

EclipseShadowReceiver::EclipseShadowReceiver(std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<core::SolarSystem> solarSystem, scene::CelestialObject const* shadowReceiver)
    : mSettings(std::move(settings))
    , mSolarSystem(std::move(solarSystem))
    , mShadowReceiver(shadowReceiver) {
}

std::string const& EclipseShadowReceiver::getShaderSnippet() {
  static std::string code(
      utils::filesystem::loadToString("../share/resources/shaders/eclipseShadows.glsl"));
  return code;
}

void EclipseShadowReceiver::init(VistaGLSLShader* shader, uint32_t textureOffset) {
  mShader        = shader;
  mTextureOffset = textureOffset;

  mUniforms.mode         = glGetUniformLocation(shader->GetProgram(), "uEclipseMode");
  mUniforms.sun          = glGetUniformLocation(shader->GetProgram(), "uEclipseSun");
  mUniforms.numOccluders = glGetUniformLocation(shader->GetProgram(), "uEclipseNumOccluders");
  mUniforms.occluders    = glGetUniformLocation(shader->GetProgram(), "uEclipseOccluders");
  mUniforms.shadowMaps   = glGetUniformLocation(shader->GetProgram(), "uEclipseShadowMaps");
}

void EclipseShadowReceiver::update(double time, scene::CelestialObserver const& observer) {
  mShadowMaps = mSolarSystem->getEclipseShadowMaps(time, *mShadowReceiver);

  for (size_t i(0); i < mShadowMaps.size() && i < MAX_BODIES; ++i) {
    scene::CelestialAnchor anchor;
    mSettings->initAnchor(anchor, mShadowMaps[i]->mCasterAnchor);
    auto pos = observer.getRelativePosition(time, anchor);

    // TODO: Can we make eclipses work with ellipsoidal casters?
    mOccluders[i] = glm::vec4(pos,
        mSettings->getAnchorRadii(mShadowMaps[i]->mCasterAnchor)[0] / observer.getAnchorScale());
  }
}

void EclipseShadowReceiver::preRender() const {

  if (mShadowMaps.size() > 0) {
    mShader->SetUniform(
        mUniforms.mode, static_cast<int>(mSettings->mGraphics.pEclipseShadowMode.get()));

    std::array<int, MAX_BODIES> shadowMapBindings{};

    for (size_t i(0); i < mShadowMaps.size() && i < MAX_BODIES; ++i) {
      shadowMapBindings[i] = static_cast<int>(i + mTextureOffset);
      mShadowMaps[i]->mTexture->Bind(GL_TEXTURE0 + shadowMapBindings[i]);
    }

    auto sunPos = mSolarSystem->pSunPosition.get();
    auto sunRadius =
        mSolarSystem->getBody("Sun")->getRadii()[0] / mSolarSystem->getObserver().getAnchorScale();
    mShader->SetUniform(mUniforms.sun, static_cast<float>(sunPos.x), static_cast<float>(sunPos.y),
        static_cast<float>(sunPos.z), static_cast<float>(sunRadius));

    mShader->SetUniform(mUniforms.numOccluders, static_cast<int>(mShadowMaps.size()));

    glUniform4fv(mUniforms.occluders, MAX_BODIES, glm::value_ptr(mOccluders[0]));
    glUniform1iv(mUniforms.shadowMaps, MAX_BODIES, shadowMapBindings.data());
  } else {
    mShader->SetUniform(mUniforms.mode, static_cast<int>(EclipseShadowMode::eNone));
  }
}

void EclipseShadowReceiver::postRender() const {
  for (size_t i = 0; i < mShadowMaps.size(); ++i) {
    mShadowMaps[i]->mTexture->Unbind(GL_TEXTURE0 + static_cast<int>(i + mTextureOffset));
  }
}

} // namespace cs::core