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
uniform vec3 uSunDirection;
uniform vec2 uRadii;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

// inputs
layout(location = 0) in vec2 iGridPos;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;
const float PI = 3.141592654;

void main()
{
    vTexCoords = iGridPos.yx;

    vec2 vDir = vec2(sin(iGridPos.x * 2.0 * PI), cos(iGridPos.x * 2.0 * PI));
    
    vec2 vPos = mix(vDir * uRadii.x, vDir * uRadii.y, iGridPos.y);

    vPosition   = (uMatModelView * vec4(vPos.x, 0, vPos.y, 1.0)).xyz;
    gl_Position =  uMatProjection * vec4(vPosition, 1);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Ring::SPHERE_FRAG = R"(
uniform sampler2D uSurfaceTexture;
uniform float uSunIlluminance;
uniform float uFarClip;

// inputs
in vec2 vTexCoords;
in vec3 vSunDirection;
in vec3 vPosition;

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

    #ifdef ENABLE_HDR
      oColor.rgb = SRGBtoLINEAR(oColor.rgb) * uSunIlluminance / M_PI;
    #else
      oColor.rgb = oColor.rgb * uSunIlluminance;
    #endif

    gl_FragDepth = length(vPosition) / uFarClip;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

Ring::Ring(std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string const& anchorName)
    : mSettings(std::move(settings))
    , mSolarSystem(std::move(solarSystem)) {

  mSettings->initAnchor(*this, anchorName);

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

void Ring::setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun) {
  mSun = sun;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Ring::Do() {
  if (!getIsInExistence() || !pVisible.get()) {
    return true;
  }

  cs::utils::FrameTimings::ScopedTimer timer("Rings");

  // (Re-)Create ring shader if necessary.
  if (mShaderDirty) {
    mShader = VistaGLSLShader();

    std::string defines = "#version 330\n";

    if (mSettings->mGraphics.pEnableHDR.get()) {
      defines += "#define ENABLE_HDR\n";
    }

    mShader.InitVertexShaderFromString(defines + SPHERE_VERT);
    mShader.InitFragmentShaderFromString(defines + SPHERE_FRAG);
    mShader.Link();

    mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
    mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
    mUniforms.surfaceTexture   = mShader.GetUniformLocation("uSurfaceTexture");
    mUniforms.radii            = mShader.GetUniformLocation("uRadii");
    mUniforms.farClip          = mShader.GetUniformLocation("uFarClip");
    mUniforms.sunIlluminance   = mShader.GetUniformLocation("uSunIlluminance");

    mShaderDirty = false;
  }

  mShader.Bind();

  // Get modelview and projection matrices.
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matMV = glm::make_mat4x4(glMatMV.data()) * glm::mat4(getWorldTransform());

  // Set uniforms.
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());

  mShader.SetUniform(mUniforms.surfaceTexture, 0);
  mShader.SetUniform(mUniforms.radii, mRingSettings.mInnerRadius, mRingSettings.mOuterRadius);
  mShader.SetUniform(mUniforms.farClip, cs::utils::getCurrentFarClipDistance());

  float sunIlluminance(1.F);

  // If HDR is enabled, the illuminance has to be calculated based on the scene's scale and the
  // distance to the Sun.
  if (mSettings->mGraphics.pEnableHDR.get()) {
    sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(getWorldTransform()[3]));
  }

  mShader.SetUniform(mUniforms.sunIlluminance, sunIlluminance);

  mTexture->Bind(GL_TEXTURE0);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Draw.
  mSphereVAO.Bind();
  glDrawArrays(GL_TRIANGLE_STRIP, 0, GRID_RESOLUTION * 2);
  mSphereVAO.Release();

  // Clean up.
  mTexture->Unbind(GL_TEXTURE0);

  glDisable(GL_BLEND);
  mShader.Release();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Ring::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::rings
