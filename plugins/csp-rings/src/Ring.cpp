////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Ring.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace csp::rings {

////////////////////////////////////////////////////////////////////////////////////////////////////

const size_t GRID_RESOLUTION = 200;

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Ring::SPHERE_VERT = R"(
uniform vec2 uRadii;
uniform mat4 uMatModel;
uniform mat4 uMatView;
uniform mat4 uMatProjection;

// inputs
layout(location = 0) in vec2 iGridPos;

// outputs
out vec2 vTexCoords;
out vec4 vPosition;

const float PI = 3.141592654;

void main() {
  vTexCoords = iGridPos.yx;

  vec2 vDir = vec2(sin(iGridPos.x * 2.0 * PI), cos(iGridPos.x * 2.0 * PI));
  vec2 vPos = mix(vDir * uRadii.x, vDir * uRadii.y, iGridPos.y);

  vPosition   = uMatModel * vec4(vPos.x, 0, vPos.y, 1.0);
  gl_Position =  uMatProjection * uMatView * vPosition;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Ring::SPHERE_FRAG = R"(
uniform sampler2D uSurfaceTexture;
uniform float uAmbientBrightness;
uniform float uSunIlluminance;
uniform float uLitSideVisible;

ECLIPSE_SHADER_SNIPPET

// inputs
in vec2 vTexCoords;
in vec4 vPosition;

// outputs
layout(location = 0) out vec4 oColor;

const float M_PI = 3.141592653589793;

vec3 SRGBtoLINEAR(vec3 srgbIn) {
  vec3 bLess = step(vec3(0.04045), srgbIn);
  return mix(srgbIn / vec3(12.92), pow((srgbIn + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
}

void main() {
  oColor = texture(uSurfaceTexture, vTexCoords);

  #ifdef ENABLE_HDR
    oColor.rgb = SRGBtoLINEAR(oColor.rgb) * uSunIlluminance / M_PI;
  #else
    oColor.rgb = oColor.rgb * uSunIlluminance;
  #endif

  #ifdef ENABLE_LIGHTING
    if (uLitSideVisible < 0.5) {
      // We darken the dark side in HDR mode, because it looks too bright compared to the planet and
      // we brighten the dark side otherwise, because it looks too dark.
      #ifdef ENABLE_HDR
        oColor.rgb = oColor.rgb * (1 - oColor.a) * 0.5;
      #else
        oColor.rgb = oColor.rgb * (1 - oColor.a) * 2.0;
      #endif
    }
    oColor.rgb = oColor.rgb * getEclipseShadow(vPosition.xyz) + vec3(uAmbientBrightness);
  #endif
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

Ring::Ring(std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string objectName)
    : mObjectName(std::move(objectName))
    , mSettings(std::move(settings))
    , mSolarSystem(std::move(solarSystem))
    , mEclipseShadowReceiver(mSettings, mSolarSystem, true) {

  // The geometry is a grid strip around the center of the SPICE frame.
  std::vector<glm::vec2> vertices(GRID_RESOLUTION * 2);

  for (size_t i = 0; i < GRID_RESOLUTION; ++i) {
    auto x = (1.F * i / (GRID_RESOLUTION - 1.F));

    vertices[i * 2 + 0] = glm::vec2(x, 0.F);
    vertices[i * 2 + 1] = glm::vec2(x, 1.F);
  }

  mSphereVBO.Bind(GL_ARRAY_BUFFER);
  mSphereVBO.BufferData(vertices.size() * sizeof(glm::vec2), vertices.data(), GL_STATIC_DRAW);
  mSphereVBO.Release();

  mSphereVAO.EnableAttributeArray(0);
  mSphereVAO.SpecifyAttributeArrayFloat(
      0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0, &mSphereVBO);

  // Recreate the shader if HDR rendering mode is toggled.
  mEnableLightingConnection = mSettings->mGraphics.pEnableLighting.connect(
      [this](bool /*enabled*/) { mShaderDirty = true; });
  mEnableHDRConnection =
      mSettings->mGraphics.pEnableHDR.connect([this](bool /*enabled*/) { mShaderDirty = true; });

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eAtmospheres) + 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Ring::~Ring() {
  mSettings->mGraphics.pEnableLighting.disconnect(mEnableLightingConnection);
  mSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Ring::configure(Plugin::Settings::Ring const& settings) {
  if (mRingSettings.mTexture != settings.mTexture) {
    mTexture = cs::graphics::TextureLoader::loadFromFile(settings.mTexture);
  }
  mRingSettings = settings;

  // Set radius for visibility culling.
  setRadii(glm::dvec3(mRingSettings.mOuterRadius));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Ring::update() {
  auto object = mSolarSystem->getObject(mObjectName);

  if (object && object->getIsBodyVisible()) {
    mEclipseShadowReceiver.update(*object);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Ring::Do() {
  auto object = mSolarSystem->getObject(mObjectName);
  if (!object || !object->getIsBodyVisible()) {
    return true;
  }

  cs::utils::FrameTimings::ScopedTimer timer("Rings");

  // (Re-)Create ring shader if necessary.
  if (mShaderDirty || mEclipseShadowReceiver.needsRecompilation()) {
    mShader = VistaGLSLShader();

    std::string defines = "#version 330\n";

    if (mSettings->mGraphics.pEnableHDR.get()) {
      defines += "#define ENABLE_HDR\n";
    }

    if (mSettings->mGraphics.pEnableLighting.get()) {
      defines += "#define ENABLE_LIGHTING\n";
    }

    std::string frag = defines + SPHERE_FRAG;
    cs::utils::replaceString(
        frag, "ECLIPSE_SHADER_SNIPPET", mEclipseShadowReceiver.getShaderSnippet());

    mShader.InitVertexShaderFromString(defines + SPHERE_VERT);
    mShader.InitFragmentShaderFromString(frag);
    mShader.Link();

    mUniforms.modelMatrix       = mShader.GetUniformLocation("uMatModel");
    mUniforms.viewMatrix        = mShader.GetUniformLocation("uMatView");
    mUniforms.projectionMatrix  = mShader.GetUniformLocation("uMatProjection");
    mUniforms.surfaceTexture    = mShader.GetUniformLocation("uSurfaceTexture");
    mUniforms.radii             = mShader.GetUniformLocation("uRadii");
    mUniforms.sunIlluminance    = mShader.GetUniformLocation("uSunIlluminance");
    mUniforms.ambientBrightness = mShader.GetUniformLocation("uAmbientBrightness");
    mUniforms.litSideVisible    = mShader.GetUniformLocation("uLitSideVisible");

    // We bind the eclipse shadow map to texture unit 1.
    mEclipseShadowReceiver.init(&mShader, 1);

    mShaderDirty = false;
  }

  mShader.Bind();

  // Get modelview and projection matrices.
  std::array<GLfloat, 16> glMatV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  glm::mat4 matM = object->getObserverRelativeTransform();
  glm::mat4 matV = glm::make_mat4x4(glMatV.data());

  // Set uniforms.
  glUniformMatrix4fv(mUniforms.modelMatrix, 1, GL_FALSE, glm::value_ptr(matM));
  glUniformMatrix4fv(mUniforms.viewMatrix, 1, GL_FALSE, glm::value_ptr(matV));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());

  mShader.SetUniform(mUniforms.surfaceTexture, 0);
  mShader.SetUniform(mUniforms.radii, mRingSettings.mInnerRadius, mRingSettings.mOuterRadius);

  float sunIlluminance(1.F);
  float ambientBrightness(mSettings->mGraphics.pAmbientBrightness.get());

  // If HDR is enabled, the illuminance has to be calculated based on the scene's scale and the
  // distance to the Sun.
  if (mSettings->mGraphics.pEnableHDR.get()) {
    sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(matM[3]));
  }

  // The following section calculates if the dark side of the ring is visible. Since the ring is a
  // flat disk we can do it before the shader. All fragments should either display the light side
  // or the dark side exclusively.
  // The check, to see which side is visible is rather simple. If the Sun and the observer look at
  // northern side of the disk, we show the light side, if the Sun and the observer look at the
  // southern side of the disk we also show the light side. In all other cases, we show the dark
  // side.
  //
  //      \ /                                o o o
  //    -- O --                          o           o                           o.O
  //      | \                          o               o
  //                         -------  o                 o  -------
  //                                   o               o
  //                                     o           o
  //                                         o o o

  glm::vec3 sunDirection =
      glm::inverse(matM) * glm::dvec4(mSolarSystem->getSunDirection(matM[3]), 0.0);

  // The dot product is positive, if the Sun is on the northern side of the ring, otherwise it is
  // negative.
  float sunAngle = glm::dot(sunDirection, glm::vec3(0.0F, 1.0F, 0.0F));

  // Some calculations to get the view and plane normal in view space.
  glm::mat4 matModelView    = matV * matM;
  glm::mat4 matNormalMatrix = glm::transpose(glm::inverse(matModelView));
  glm::vec4 viewPos         = matModelView * glm::vec4(0.0F, 0.0F, 0.0F, 1.0F);
  viewPos                   = viewPos / viewPos.w;
  glm::vec3 planeNormal     = glm::normalize(-viewPos.xyz());
  glm::vec3 viewNormal      = (matNormalMatrix * glm::vec4(0.0F, 1.0F, 0.0F, 0.0F)).xyz();

  // The dot product is positive, if the observer is on the northern side of the ring, otherwise it
  // is negative.
  float viewAngle = glm::dot(planeNormal, viewNormal);

  // When Sun and observer are on the same side of the disk it is lit, otherwise it is dark.
  float litSideVisible =
      (sunAngle > 0.0F && viewAngle > 0.0F) || (sunAngle < 0.0F && viewAngle < 0.0F) ? 1.0F : 0.0F;

  mShader.SetUniform(mUniforms.sunIlluminance, sunIlluminance);
  mShader.SetUniform(mUniforms.ambientBrightness, ambientBrightness);
  mShader.SetUniform(mUniforms.litSideVisible, litSideVisible);

  mTexture->Bind(GL_TEXTURE0);

  glEnable(GL_BLEND);
  glDisable(GL_CULL_FACE);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Initialize eclipse shadow-related uniforms and textures.
  mEclipseShadowReceiver.preRender();

  // Draw.
  mSphereVAO.Bind();
  glDrawArrays(GL_TRIANGLE_STRIP, 0, GRID_RESOLUTION * 2);
  mSphereVAO.Release();

  // Reset eclipse shadow-related texture units.
  mEclipseShadowReceiver.postRender();

  // Clean up.
  mTexture->Unbind(GL_TEXTURE0);

  glDisable(GL_BLEND);
  glEnable(GL_CULL_FACE);

  mShader.Release();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Ring::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::rings
