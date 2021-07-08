////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SimpleBody.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
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

const char* Hemisphere::SPHERE_VERT = R"(
uniform vec3 uSunDirection;
uniform vec3 uRadii;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

// inputs
layout(location = 0) in vec2 iGridPos;

// outputs
out vec2 vTexCoords;
out vec3 vNormal;
out vec3 vPosition;
out vec3 vCenter;
out vec2 vLngLat;

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

void main()
{
    vTexCoords = vec2(iGridPos.x, 1-iGridPos.y);
    vLngLat.x = iGridPos.x * 2.0 * PI - PI;
    vLngLat.y = iGridPos.y * PI - PI/2;
    vPosition = toCartesian(vLngLat);
    vNormal    = (uMatModelView * vec4(geodeticSurfaceNormal(vLngLat), 0.0)).xyz;
    vPosition   = (uMatModelView * vec4(vPosition, 1.0)).xyz;
    vCenter     = (uMatModelView * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    gl_Position =  uMatProjection * vec4(vPosition, 1);

    if (gl_Position.w > 0) {
      gl_Position /= gl_Position.w;
      if (gl_Position.z >= 1) {
        gl_Position.z = 0.999999;
      }
    }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Hemisphere::SPHERE_FRAG = R"(
uniform vec3 uSunDirection;
uniform sampler2D uSurfaceTexture;
uniform float uAmbientBrightness;
uniform float uSunIlluminance;
uniform float uFarClip;

// inputs
in vec2 vTexCoords;
in vec3 vNormal;
in vec3 vPosition;
in vec3 vCenter;
in vec2 vLngLat;

// outputs
layout(location = 0) out vec4 oColor;

const float M_PI = 3.141592653589793;

vec3 SRGBtoLINEAR(vec3 srgbIn)
{
  vec3 bLess = step(vec3(0.04045),srgbIn);
  return mix( srgbIn/vec3(12.92), pow((srgbIn+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
}
    
void main()
{
    oColor = texture(uSurfaceTexture, vTexCoords);
    vec3 color = oColor.rgb;

    #ifdef ENABLE_HDR
      color = SRGBtoLINEAR(color.rgb) * uSunIlluminance / M_PI;
    #else
      color = color.rgb * uSunIlluminance;
    #endif

    #ifdef ENABLE_LIGHTING
      vec3 normal = normalize(vNormal);
      float light = max(dot(normal, uSunDirection), 0.0);
      if (gl_FrontFacing) {
        light *= 0.3;
        color = mix(color*uAmbientBrightness*0.5, color, light);
      } else {
        color = mix(color*uAmbientBrightness, color, light);
      }
    #endif
    oColor.rgb = color;

    gl_FragDepth = length(vPosition) / uFarClip;
    if (gl_FrontFacing) {
      oColor = vec4(0, 0, 0, 1);
    }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

Hemisphere::Hemisphere(bool front, std::shared_ptr<cs::core::Settings> settings,
    Plugin::Settings& pluginSettings, std::shared_ptr<cs::core::SolarSystem> solarSystem,
    SimpleBody const& parent)
    : mParent(parent)
    , mFront(front)
    , mSettings(std::move(settings))
    , mPluginSettings(pluginSettings)
    , mSolarSystem(std::move(solarSystem)) {
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
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems) + (mFront ? 20 : 0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SimpleBody::SimpleBody(std::shared_ptr<cs::core::Settings> settings,
    Plugin::Settings& pluginSettings, std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::string const& anchorName)
    : mFrontHemisphere(true, settings, pluginSettings, solarSystem, *this)
    , mBackHemisphere(false, settings, pluginSettings, solarSystem, *this) {
  settings->initAnchor(*this, anchorName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Hemisphere::~Hemisphere() {
  mSettings->mGraphics.pEnableLighting.disconnect(mEnableLightingConnection);
  mSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Hemisphere::configure(Plugin::Settings::SimpleBody const& settings) {
  mTexture = cs::graphics::TextureLoader::loadFromFile(settings.mTexture);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SimpleBody::configure(Plugin::Settings::SimpleBody const& settings) {
  if (mSimpleBodySettings.mTexture != settings.mTexture) {
    mFrontHemisphere.configure(settings);
    mBackHemisphere.configure(settings);
  }
  mSimpleBodySettings = settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Hemisphere::setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun) {
  mSun = sun;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SimpleBody::setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun) {
  mFrontHemisphere.setSun(sun);
  mBackHemisphere.setSun(sun);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SimpleBody::getIntersection(
    glm::dvec3 const& rayOrigin, glm::dvec3 const& rayDir, glm::dvec3& pos) const {

  auto invTransform = glm::inverse(getWorldTransform());

  // Transform ray into planet coordinate system.
  glm::dvec4 origin(rayOrigin, 1.0);
  origin = (invTransform * origin) / glm::dvec4(mRadii, 1.0);

  glm::dvec4 direction(rayDir, 0.0);
  direction = (invTransform * direction) / glm::dvec4(mRadii, 1.0);
  direction = glm::normalize(direction);

  double b    = glm::dot(origin.xyz(), direction.xyz());
  double c    = glm::dot(origin.xyz(), origin.xyz()) - 1.0;
  double fDet = b * b - c;

  if (fDet < 0.0) {
    return false;
  }

  fDet = std::sqrt(fDet);
  pos  = (origin + direction * (-b - fDet));
  pos *= mRadii;

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double SimpleBody::getHeight(glm::dvec2 /*lngLat*/) const {
  // This is why we call them 'SimpleBodies'.
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Hemisphere::Do() {
  if (!mParent.getIsInExistence() || !mParent.pVisible.get() || !mPluginSettings.mEnabled.get()) {
    return true;
  }

  cs::utils::FrameTimings::ScopedTimer timer("Simple Bodies");

  if (mShaderDirty) {
    mShader = VistaGLSLShader();

    // (Re-)create sphere shader.
    std::string defines = "#version 330\n";

    if (mSettings->mGraphics.pEnableHDR.get()) {
      defines += "#define ENABLE_HDR\n";
    }

    if (mSettings->mGraphics.pEnableLighting.get()) {
      defines += "#define ENABLE_LIGHTING\n";
    }

    mShader.InitVertexShaderFromString(defines + SPHERE_VERT);
    mShader.InitFragmentShaderFromString(defines + SPHERE_FRAG);
    mShader.Link();

    mUniforms.sunDirection      = mShader.GetUniformLocation("uSunDirection");
    mUniforms.sunIlluminance    = mShader.GetUniformLocation("uSunIlluminance");
    mUniforms.ambientBrightness = mShader.GetUniformLocation("uAmbientBrightness");
    mUniforms.modelViewMatrix   = mShader.GetUniformLocation("uMatModelView");
    mUniforms.projectionMatrix  = mShader.GetUniformLocation("uMatProjection");
    mUniforms.surfaceTexture    = mShader.GetUniformLocation("uSurfaceTexture");
    mUniforms.radii             = mShader.GetUniformLocation("uRadii");
    mUniforms.farClip           = mShader.GetUniformLocation("uFarClip");

    mShaderDirty = false;
  }

  mShader.Bind();

  glm::vec3 sunDirection(1, 0, 0);
  float     sunIlluminance(1.F);
  float     ambientBrightness(mSettings->mGraphics.pAmbientBrightness.get());

  if (mParent.getCenterName() == "Sun") {
    // If the SimpleBody is actually the sun, we have to calculate the lighting differently.
    if (mSettings->mGraphics.pEnableHDR.get()) {
      double sceneScale = 1.0 / mSolarSystem->getObserver().getAnchorScale();

      // To get the luminous exitance (in lux) of the Sun, we have to divide its luminous power (in
      // lumens) by its surface area.
      double luminousExitance =
          mSolarSystem->pSunLuminousPower.get() / (sceneScale * sceneScale * mParent.mRadii[0] *
                                                      mParent.mRadii[0] * 4.0 * glm::pi<double>());

      // We consider the Sun to emit light equally in all directions. So we have to divide the
      // luminous exitance by PI to get actual luminance values.
      double sunLuminance = luminousExitance / glm::pi<double>();

      // The variable is called illuminance, for the sun it contains actually luminance values.
      sunIlluminance = static_cast<float>(sunLuminance);
    }

    ambientBrightness = 1.0F;

  } else if (mSun) {
    // For all other bodies we can use the utility methods from the SolarSystem.
    if (mSettings->mGraphics.pEnableHDR.get()) {
      sunIlluminance =
          static_cast<float>(mSolarSystem->getSunIlluminance(mParent.getWorldTransform()[3]));
    }

    sunDirection = mSolarSystem->getSunDirection(mParent.getWorldTransform()[3]);
  }

  mShader.SetUniform(mUniforms.sunDirection, sunDirection[0], sunDirection[1], sunDirection[2]);
  mShader.SetUniform(mUniforms.sunIlluminance, sunIlluminance);
  mShader.SetUniform(mUniforms.ambientBrightness, ambientBrightness);

  // Get modelview and projection matrices.
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matMV = glm::make_mat4x4(glMatMV.data()) * glm::mat4(mParent.getWorldTransform());
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());

  mShader.SetUniform(mUniforms.surfaceTexture, 0);
  mShader.SetUniform(mUniforms.radii, static_cast<float>(mParent.mRadii[0]),
      static_cast<float>(mParent.mRadii[1]), static_cast<float>(mParent.mRadii[2]));
  mShader.SetUniform(mUniforms.farClip, cs::utils::getCurrentFarClipDistance());

  mTexture->Bind(GL_TEXTURE0);

  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_CULL_FACE);
  glCullFace(mFront ? GL_FRONT : GL_BACK);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Draw.
  mSphereVAO.Bind();
  glDrawElements(GL_TRIANGLE_STRIP, (GRID_RESOLUTION_X - 1) * (2 + 2 * GRID_RESOLUTION_Y),
      GL_UNSIGNED_INT, nullptr);
  mSphereVAO.Release();

  // Clean up.
  mTexture->Unbind(GL_TEXTURE0);
  mShader.Release();
  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Hemisphere::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::simplebodies
