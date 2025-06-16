////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SimpleBody.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace csp::simplebodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

const uint32_t GRID_RESOLUTION_X = 200;
const uint32_t GRID_RESOLUTION_Y = 100;

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* SimpleBody::SPHERE_VERT = R"(
uniform vec3 uSunDirection;
uniform vec3 uRadii;
uniform mat4 uMatModel;
uniform mat4 uMatView;
uniform mat4 uMatProjection;

// inputs
layout(location = 0) in vec2 iGridPos;

// outputs
out vec2 vTexCoords;
out vec3 vNormal;
out vec3 vPosition;
out vec3 vCenter;
out vec3 vNorth;
out vec2 vLngLat;
out vec3 vSunDirection;

const float PI = 3.141592654;

vec3 geodeticSurfaceNormal(vec2 lngLat) {
  return vec3(cos(lngLat.y) * sin(lngLat.x), sin(lngLat.y),
      cos(lngLat.y) * cos(lngLat.x));
}

vec3 toCartesian(vec2 lonLat) {
  vec3 n = geodeticSurfaceNormal(lonLat);
  vec3 k = n * uRadii * uRadii;
  float gamma = sqrt(dot(k, n));
  return k / gamma;
}

void main() {
#ifdef PRIME_MERIDIAN_IN_CENTER
  vTexCoords = vec2(iGridPos.x, 1 - iGridPos.y);
#else
  vTexCoords = vec2(iGridPos.x + 0.5, 1 - iGridPos.y);
#endif
  vLngLat.x = iGridPos.x * 2.0 * PI - PI;
  vLngLat.y = iGridPos.y * PI - PI/2;
  vPosition = toCartesian(vLngLat);
  vNormal       = (uMatModel * vec4(geodeticSurfaceNormal(vLngLat), 0.0)).xyz;
  vPosition     = (uMatModel * vec4(vPosition, 1.0)).xyz;
  vNorth        = (uMatModel * vec4(0.0, 1.0, 0.0, 0.0)).xyz;
  vSunDirection = (uMatModel * vec4(uSunDirection, 0.0)).xyz;
  vCenter       = (uMatModel * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
  gl_Position   = uMatProjection * uMatView * vec4(vPosition, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* SimpleBody::SPHERE_FRAG = R"(
uniform sampler2D uSurfaceTexture;
uniform float uAmbientBrightness;
uniform float uSunIlluminance;

#ifdef HAS_RING
  uniform sampler2D uRingTexture;
  uniform vec2 uRingRadii;
#endif

ECLIPSE_SHADER_SNIPPET

// inputs
in vec2 vTexCoords;
in vec3 vNormal;
in vec3 vPosition;
in vec3 vCenter;
in vec3 vNorth;
in vec2 vLngLat;
in vec3 vSunDirection;

// outputs
layout(location = 0) out vec3 oColor;

const float PI = 3.141592654;
const float E  = 2.718281828;

vec3 SRGBtoLINEAR(vec3 srgbIn) {
  vec3 bLess = step(vec3(0.04045),srgbIn);
  return mix( srgbIn/vec3(12.92), pow((srgbIn+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
}

// Calculates the shading by planetary rings. This is done in 3 steps:
// 1. Calculate the intersection between a ray from the fragment to the Sun and the ring plane.
// 2. If an intersection exists check if it falls within the ring.
// 3. If it falls within the ring, we get the brightness from the ring texture.
//
//                 o o o                   \ /
//             o           o             -- O --
//           o               o             / \
// -------  o                 o  -------
//           o               o
//             o           o
//                 o o o
//
float getRingShadow() {
  #ifdef HAS_RING
    // The up direction of the ring plane.
    vec3 ringNormal = normalize(vNorth);

    // The normalized direction of the sun.
    vec3 sunNormal = normalize(vSunDirection);

    // If the ray is parallel to the ring, we don't draw a shadow.
    float sunAngle = dot(ringNormal, sunNormal);
    if (sunAngle == 0) {
      return 1.0;
    }

    // The distance along the ray from the fragment to the intersection.
    float t = (dot(ringNormal, vCenter) - dot(ringNormal, vPosition)) / dot(ringNormal, sunNormal);

    // If the distance is negative, the ray intersects the ring away from the Sun.
    if (t < 0.0) {
      return 1.0;
    }

    // The exact point of intersection with the ring plane.
    vec3 intersect = vPosition + (sunNormal * t);

    // The distance from the bodies center. If it is outside the ring radii, we don't have a shadow.
    float dist = length(vCenter - intersect);
    if (dist < uRingRadii.x || dist > uRingRadii.y) {
      return 1.0;
    }

    // Convert the distance to the texture coordinate.
    float texPosition = (dist - uRingRadii.x) / (uRingRadii.y - uRingRadii.x);
    return 1.0 - texture(uRingTexture, vec2(texPosition, 0.5)).a;
  #else
    return 1.0;
  #endif
}

// placeholder for the BRDF in HDR mode
$BRDF_HDR

// placeholder for the BRDF in light mode
$BRDF_NON_HDR

void main() {
    oColor = texture(uSurfaceTexture, vTexCoords).rgb;

    // Needed for the BRDFs.
    vec3 N = normalize(vNormal);
    vec3 L = normalize(vSunDirection);
    vec3 V = normalize(-vPosition);
    float cos_i = dot(N, L);
    float cos_r = dot(N, V);

#ifdef ENABLE_HDR
    // Make the amount of ambient brightness perceptually linear in HDR mode.
    float ambient = pow(uAmbientBrightness, E);
    oColor = SRGBtoLINEAR(oColor) * uSunIlluminance / $AVG_LINEAR_IMG_INTENSITY;
    float f_r = BRDF_HDR(N, L, V);
#else
    float ambient = uAmbientBrightness;
    oColor = oColor * uSunIlluminance;
    float f_r = BRDF_NON_HDR(N, L, V);
#endif

#ifdef ENABLE_LIGHTING
    vec3 light = vec3(1.0);
    light *= max(0.0, cos_i);
    if (cos_i > 0) {
      if (f_r < 0 || isnan(f_r) || isinf(f_r)) {
        light *= 0;
      } else {
        light *= f_r * getRingShadow() * getEclipseShadow(vPosition);
      }
    }
    oColor = mix(oColor * light, oColor, ambient);
#endif

// conserve energy
#if defined(ENABLE_HDR) && !defined(ENABLE_LIGHTING)
    oColor /= PI;
#endif
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

SimpleBody::SimpleBody(std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<cs::core::SolarSystem>                 solarSystem)
    : mSettings(std::move(settings))
    , mSolarSystem(std::move(solarSystem))
    , mEclipseShadowReceiver(mSettings, mSolarSystem, false) {

  // For rendering the sphere, we create a 2D-grid which is warped into a sphere in the vertex
  // shader. The vertex positions are directly used as texture coordinates.
  std::vector<float>    vertices(GRID_RESOLUTION_X * GRID_RESOLUTION_Y * 2);
  std::vector<unsigned> indices((GRID_RESOLUTION_X - 1) * (2 + 2 * GRID_RESOLUTION_Y));

  for (uint32_t x = 0; x < GRID_RESOLUTION_X; ++x) {
    for (uint32_t y = 0; y < GRID_RESOLUTION_Y; ++y) {
      vertices[(x * GRID_RESOLUTION_Y + y) * 2 + 0] = 1.F / (GRID_RESOLUTION_X - 1) * x;
      vertices[(x * GRID_RESOLUTION_Y + y) * 2 + 1] = 1.F / (GRID_RESOLUTION_Y - 1) * y;
    }
  }

  uint32_t index = 0;

  for (uint32_t x = 0; x < GRID_RESOLUTION_X - 1; ++x) {
    indices[index++] = x * GRID_RESOLUTION_Y;
    for (uint32_t y = 0; y < GRID_RESOLUTION_Y; ++y) {
      indices[index++] = x * GRID_RESOLUTION_Y + y;
      indices[index++] = (x + 1) * GRID_RESOLUTION_Y + y;
    }
    indices[index] = indices[index - 1];
    ++index;
  }

  mSphereVAO.Bind();

  mSphereVBO.Bind(GL_ARRAY_BUFFER);
  mSphereVBO.BufferData(vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  mSphereIBO.Bind(GL_ELEMENT_ARRAY_BUFFER);
  mSphereIBO.BufferData(indices.size() * sizeof(unsigned), indices.data(), GL_STATIC_DRAW);

  mSphereVAO.EnableAttributeArray(0);
  mSphereVAO.SpecifyAttributeArrayFloat(
      0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0, &mSphereVBO);

  mSphereVAO.Release();
  mSphereIBO.Release();
  mSphereVBO.Release();

  // Recreate the shader if lighting or HDR rendering mode are toggled.
  mEnableLightingConnection = mSettings->mGraphics.pEnableLighting.connect(
      [this](bool /*enabled*/) { mShaderDirty = true; });
  mEnableHDRConnection =
      mSettings->mGraphics.pEnableHDR.connect([this](bool /*enabled*/) { mShaderDirty = true; });

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::ePlanets));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SimpleBody::~SimpleBody() {
  mSettings->mGraphics.pEnableLighting.disconnect(mEnableLightingConnection);
  mSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SimpleBody::configure(Plugin::Settings::SimpleBody const& settings) {
  if (mSimpleBodySettings.mTexture != settings.mTexture) {
    mTexture = cs::graphics::TextureLoader::loadFromFile(settings.mTexture);
  }

  if (settings.mRing && mSimpleBodySettings.mRing->mTexture != settings.mRing->mTexture) {
    mRingTexture = cs::graphics::TextureLoader::loadFromFile(settings.mRing->mTexture);
  }

  if (mSimpleBodySettings.mPrimeMeridianInCenter != settings.mPrimeMeridianInCenter) {
    mShaderDirty = true;
  }

  mSimpleBodySettings = settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SimpleBody::setObjectName(std::string objectName) {
  mObjectName = std::move(objectName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& SimpleBody::getObjectName() const {
  return mObjectName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SimpleBody::getIntersection(
    glm::dvec3 const& rayOrigin, glm::dvec3 const& rayDir, glm::dvec3& pos) const {

  auto parent = mSolarSystem->getObject(mObjectName);

  if (!parent || !parent->getIsBodyVisible()) {
    return false;
  }

  auto invTransform = glm::inverse(parent->getObserverRelativeTransform());

  // Transform ray into planet coordinate system.
  glm::dvec4 origin(rayOrigin, 1.0);
  origin = (invTransform * origin) / glm::dvec4(parent->getRadii(), 1.0);

  glm::dvec4 direction(rayDir, 0.0);
  direction = (invTransform * direction) / glm::dvec4(parent->getRadii(), 1.0);
  direction = glm::normalize(direction);

  double b    = glm::dot(origin.xyz(), direction.xyz());
  double c    = glm::dot(origin.xyz(), origin.xyz()) - 1.0;
  double fDet = b * b - c;

  if (fDet < 0.0) {
    return false;
  }

  fDet = std::sqrt(fDet);
  pos  = (origin + direction * (-b - fDet));
  pos *= parent->getRadii();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double SimpleBody::getHeight(glm::dvec2 /*lngLat*/) const {
  // This is why we call them 'SimpleBodies'.
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SimpleBody::update() {

  auto parent = mSolarSystem->getObject(mObjectName);

  if (parent && parent->getIsBodyVisible()) {
    cs::utils::FrameStats::ScopedTimer timer(
        "Update " + parent->getCenterName(), cs::utils::FrameStats::TimerMode::eCPU);
    mEclipseShadowReceiver.update(*parent);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SimpleBody::Do() {
  auto parent = mSolarSystem->getObject(mObjectName);

  if (!parent || !parent->getIsBodyVisible()) {
    return true;
  }

  cs::utils::FrameStats::ScopedTimer timer("Draw " + parent->getCenterName());

  if (mShaderDirty || mEclipseShadowReceiver.needsRecompilation()) {
    mShader = VistaGLSLShader();

    // (Re-)create sphere shader.
    std::string defines = "#version 430\n";

    if (mSettings->mGraphics.pEnableHDR.get()) {
      defines += "#define ENABLE_HDR\n";
    }

    if (mSettings->mGraphics.pEnableLighting.get()) {
      defines += "#define ENABLE_LIGHTING\n";
    }

    if (mSimpleBodySettings.mPrimeMeridianInCenter.get()) {
      defines += "#define PRIME_MERIDIAN_IN_CENTER\n";
    }

    if (mSimpleBodySettings.mRing) {
      defines += "#define HAS_RING\n";
    }

    std::string vert = defines + SPHERE_VERT;
    std::string frag = defines + SPHERE_FRAG;

    cs::utils::replaceString(
        frag, "ECLIPSE_SHADER_SNIPPET", mEclipseShadowReceiver.getShaderSnippet());

    cs::core::Settings::Shading const& shading = mSettings.get()->getShadingForBody(mObjectName);

    std::string brdfHdrSnippet    = shading.pBrdfHdr.get().assembleShaderSnippet("BRDF_HDR");
    std::string brdfNonHdrSnippet = shading.pBrdfNonHdr.get().assembleShaderSnippet("BRDF_NON_HDR");
    float       avgLinearImgIntensity = shading.pAvgLinearImgIntensity.get();

    // Inject correct identifiers so the fragment shader can find the functions;
    // inject the functions in the fragment shader
    cs::utils::replaceString(frag, "$BRDF_HDR", brdfHdrSnippet);
    cs::utils::replaceString(frag, "$BRDF_NON_HDR", brdfNonHdrSnippet);
    cs::utils::replaceString(
        frag, "$AVG_LINEAR_IMG_INTENSITY", std::to_string(avgLinearImgIntensity));

    mShader.InitVertexShaderFromString(vert);
    mShader.InitFragmentShaderFromString(frag);
    mShader.Link();

    mUniforms.sunDirection      = mShader.GetUniformLocation("uSunDirection");
    mUniforms.sunIlluminance    = mShader.GetUniformLocation("uSunIlluminance");
    mUniforms.ambientBrightness = mShader.GetUniformLocation("uAmbientBrightness");
    mUniforms.modelMatrix       = mShader.GetUniformLocation("uMatModel");
    mUniforms.viewMatrix        = mShader.GetUniformLocation("uMatView");
    mUniforms.projectionMatrix  = mShader.GetUniformLocation("uMatProjection");
    mUniforms.surfaceTexture    = mShader.GetUniformLocation("uSurfaceTexture");
    mUniforms.radii             = mShader.GetUniformLocation("uRadii");

    if (mSimpleBodySettings.mRing) {
      mUniforms.ringTexture = mShader.GetUniformLocation("uRingTexture");
      mUniforms.ringRadii   = mShader.GetUniformLocation("uRingRadii");
    }

    // We bind the eclipse shadow map to texture unit 2.
    mEclipseShadowReceiver.init(&mShader, 2);

    mShaderDirty = false;
  }

  mShader.Bind();

  glm::vec3 sunDirection(1, 0, 0);
  float     sunIlluminance(1.F);
  float     ambientBrightness(mSettings->mGraphics.pAmbientBrightness.get());
  auto      transform = parent->getObserverRelativeTransform();

  if (parent == mSolarSystem->getSun()) {
    // If the SimpleBody is actually the sun, we have to calculate the lighting differently.
    if (mSettings->mGraphics.pEnableHDR.get()) {

      // The variable is called illuminance, for the sun it contains actually luminance values.
      sunIlluminance = static_cast<float>(mSolarSystem->getSunLuminance());

      // For planets, this illuminance is divided by pi, so we have to premultiply it for the sun.
      sunIlluminance *= glm::pi<float>();
    }

    ambientBrightness = 1.0F;

  } else {
    // For all other bodies we can use the utility methods from the SolarSystem.
    if (mSettings->mGraphics.pEnableHDR.get()) {
      sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(transform[3]));
    }

    sunDirection =
        glm::inverse(transform) * glm::dvec4(mSolarSystem->getSunDirection(transform[3]), 0.0);
  }

  mShader.SetUniform(mUniforms.sunDirection, sunDirection[0], sunDirection[1], sunDirection[2]);
  mShader.SetUniform(mUniforms.sunIlluminance, sunIlluminance);
  mShader.SetUniform(mUniforms.ambientBrightness, ambientBrightness);

  // Get modelview and projection matrices.
  std::array<GLfloat, 16> glMatV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matM = glm::mat4(transform);
  auto matV = glm::make_mat4x4(glMatV.data());
  glUniformMatrix4fv(mUniforms.modelMatrix, 1, GL_FALSE, glm::value_ptr(matM));
  glUniformMatrix4fv(mUniforms.viewMatrix, 1, GL_FALSE, glm::value_ptr(matV));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());

  glm::vec3 radii = parent->getRadii();
  mShader.SetUniform(mUniforms.radii, radii[0], radii[1], radii[2]);

  // Set the texture wrapping on the x-axis to repeat, so we can easily deal with textures, where
  // the prime meridian is not in the center.
  int32_t wrapMode = 0;
  glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, &wrapMode);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);

  mShader.SetUniform(mUniforms.surfaceTexture, 0);
  mTexture->Bind(GL_TEXTURE0);

  if (mSimpleBodySettings.mRing) {
    mShader.SetUniform(mUniforms.ringRadii,
        static_cast<float>(mSimpleBodySettings.mRing->mInnerRadius * parent->getScale() /
                           mSolarSystem->getObserver().getScale()),
        static_cast<float>(mSimpleBodySettings.mRing->mOuterRadius * parent->getScale() /
                           mSolarSystem->getObserver().getScale()));
    mShader.SetUniform(mUniforms.ringTexture, 1);
    mRingTexture->Bind(GL_TEXTURE1);
  }

  // Initialize eclipse shadow-related uniforms and textures.
  mEclipseShadowReceiver.preRender();

  // Draw.
  mSphereVAO.Bind();
  glDrawElements(GL_TRIANGLE_STRIP, (GRID_RESOLUTION_X - 1) * (2 + 2 * GRID_RESOLUTION_Y),
      GL_UNSIGNED_INT, nullptr);
  mSphereVAO.Release();

  // Reset eclipse shadow-related texture units.
  mEclipseShadowReceiver.postRender();

  // Clean up.
  mTexture->Unbind(GL_TEXTURE0);

  if (mSimpleBodySettings.mRing) {
    mRingTexture->Unbind();
  }

  mShader.Release();

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapMode);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SimpleBody::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::simplebodies
