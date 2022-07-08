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

////////////////////////////////////////////////////////////////////////////////////////////////////

EclipseShadowReceiver::EclipseShadowReceiver(std::shared_ptr<Settings> settings,
    std::shared_ptr<SolarSystem> solarSystem, scene::CelestialObject const* shadowReceiver,
    bool allowSelfShadowing)
    : mSettings(std::move(settings))
    , mSolarSystem(std::move(solarSystem))
    , mShadowReceiver(shadowReceiver)
    , mAllowSelfShadowing(allowSelfShadowing) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool EclipseShadowReceiver::needsRecompilation() const {
  return mLastEclipseShadowMode != mSettings->mGraphics.pEclipseShadowMode.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string EclipseShadowReceiver::getShaderSnippet() const {

  // Load the code only once.
  static std::string code(
      utils::filesystem::loadToString("../share/resources/shaders/eclipseShadows.glsl"));

  // Inject the current eclipse mode into a copy of the string.
  auto copy = code;
  cs::utils::replaceString(copy, "ECLIPSE_MODE",
      cs::utils::toString(static_cast<int>(mSettings->mGraphics.pEclipseShadowMode.get())));

  // Store the last use mode. This is required for needsRecompilation().
  mLastEclipseShadowMode = mSettings->mGraphics.pEclipseShadowMode.get();

  return copy;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void EclipseShadowReceiver::init(VistaGLSLShader* shader, uint32_t textureOffset) {
  mShader        = shader;
  mTextureOffset = textureOffset;

  mUniforms.sun          = glGetUniformLocation(shader->GetProgram(), "uEclipseSun");
  mUniforms.numOccluders = glGetUniformLocation(shader->GetProgram(), "uEclipseNumOccluders");
  mUniforms.occluders    = glGetUniformLocation(shader->GetProgram(), "uEclipseOccluders");
  mUniforms.shadowMaps   = glGetUniformLocation(shader->GetProgram(), "uEclipseShadowMaps");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void EclipseShadowReceiver::update(double time, scene::CelestialObserver const& observer) {

  // Acquire a list of allpotentially relevant eclipse shadow maps.
  mShadowMaps = mSolarSystem->getEclipseShadowMaps(time, *mShadowReceiver, mAllowSelfShadowing);

  // For each shadow-casting body, we store the observer-relative position and the observer-relative
  // radius. For now, all occluders are considered to be spheres.
  for (size_t i(0); i < mShadowMaps.size() && i < MAX_BODIES; ++i) {
    auto object = mSettings->getAnchor(mShadowMaps[i]->mOccluderAnchor);
    auto pos = object->getObserverRelativePosition();

    mOccluders[i] = glm::vec4(pos, object->getRadii()[0] / observer.getAnchorScale());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void EclipseShadowReceiver::preRender() const {

  mShader->SetUniform(mUniforms.numOccluders, static_cast<int>(mShadowMaps.size()));

  // Bind all eclipse shadow maps and upload the respective caster positions and radii.
  if (!mShadowMaps.empty()) {
    std::array<int, MAX_BODIES> shadowMapBindings{};

    for (size_t i(0); i < mShadowMaps.size() && i < MAX_BODIES; ++i) {
      shadowMapBindings[i] = static_cast<int>(i + mTextureOffset);
      mShadowMaps[i]->mTexture->Bind(GL_TEXTURE0 + shadowMapBindings[i]);
    }

    glUniform4fv(mUniforms.occluders, MAX_BODIES, glm::value_ptr(mOccluders[0]));
    glUniform1iv(mUniforms.shadowMaps, MAX_BODIES, shadowMapBindings.data());

    // Also, the Sun's position and radius is required.
    auto sunPos = mSolarSystem->pSunPosition.get();
    auto sunRadius =
        mSolarSystem->getBody("Sun")->getRadii()[0] / mSolarSystem->getObserver().getAnchorScale();
    mShader->SetUniform(mUniforms.sun, static_cast<float>(sunPos.x), static_cast<float>(sunPos.y),
        static_cast<float>(sunPos.z), static_cast<float>(sunRadius));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void EclipseShadowReceiver::postRender() const {
  for (size_t i = 0; i < mShadowMaps.size(); ++i) {
    mShadowMaps[i]->mTexture->Unbind(GL_TEXTURE0 + static_cast<int>(i + mTextureOffset));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core