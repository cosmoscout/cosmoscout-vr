////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Sharad.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-scene/CelestialObserver.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <glm/gtc/type_ptr.hpp>

#include <cstdio>
#include <utility>

namespace csp::sharad {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Sharad::VERT = R"(
#version 330

uniform mat4 uMatModelView;
uniform mat4 uMatProjection;
uniform float uHeightScale;
uniform vec3 uRadii;

// inputs
layout(location = 0) in vec2  iLngLat;
layout(location = 1) in vec2  iTexCoords;
layout(location = 2) in float iTime;

// outputs
out vec3  vPositionVS;
out vec4  vPositionSS;
out vec2  vTexCoords;
out float vTime;

vec3 geodeticSurfaceNormal(vec2 lngLat) {
  return vec3(cos(lngLat.y) * sin(lngLat.x), sin(lngLat.y),
      cos(lngLat.y) * cos(lngLat.x));
}

vec3 toCartesian(vec2 lonLat, float height) {
  vec3 n = geodeticSurfaceNormal(lonLat);
  vec3 k = n * uRadii * uRadii;
  float gamma = sqrt(dot(k, n));
  return k / gamma + height * n;
}

void main()
{
    vTexCoords = iTexCoords;
    vTime      = iTime;

    float height = vTexCoords.y < 0.5 ? 10000 * uHeightScale: -10100 * uHeightScale;
    vPositionVS  = (uMatModelView * vec4(toCartesian(iLngLat, height), 1.0)).xyz;
    vPositionSS  = uMatProjection * vec4(vPositionVS, 1);
    gl_Position  = vPositionSS;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Sharad::FRAG = R"(
#version 330

uniform mat4 uMatProjection;
uniform sampler2D uDepthBuffer;
uniform sampler2D uSharadTexture;
uniform float uAmbientBrightness;
uniform float uTime;
uniform float uSceneScale;
uniform vec2 uViewportPos;

// inputs
in vec3  vPositionVS;
in vec4  vPositionSS;
in vec2  vTexCoords;
in float vTime;

// outputs
layout(location = 0) out vec4 oColor;

void main()
{
    if (vTime > uTime)
    {
        discard;
    }

    float fDepth = texture(uDepthBuffer, (gl_FragCoord.xy - uViewportPos) / textureSize(uDepthBuffer, 0)).r;
    vec4 surfacePos = inverse(uMatProjection) * vec4(vPositionSS.xy / vPositionSS.w, 2*fDepth-1, 1);
    float surfaceDistance = length(surfacePos.xyz / surfacePos.w);
    float sharadDistance  = length(vPositionVS);
    
    if (sharadDistance < surfaceDistance)
    {
        discard;
    }

    float val = texture(uSharadTexture, vTexCoords).r;
    val = mix(1, val, clamp((uTime - vTime), 0, 1));

    oColor.r = pow(val,  0.5);
    oColor.g = pow(val,  2.0);
    oColor.b = pow(val, 10.0);
    oColor.a = 1.0 - clamp((sharadDistance - surfaceDistance) * uSceneScale / 30000, 0.1, 1.0);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ProfileRadarData {
  unsigned int Number;
  unsigned int Year;
  unsigned int Month;
  unsigned int Day;
  unsigned int Hour;
  unsigned int Minute;
  unsigned int Second;
  unsigned int Millisecond;
  float        Latitude;
  float        Longitude;
  float        SurfaceAltitude;
  float        MROAltitude;
  float        c;
  float        d;
  float        e;
  float        f;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

Sharad::Sharad(std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<cs::core::GraphicsEngine>      graphicsEngine,
    std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string objectName,
    std::string const& sTiffFile, std::string const& sTabFile)
    : mSettings(std::move(settings))
    , mGraphicsEngine(std::move(graphicsEngine))
    , mSolarSystem(std::move(solarSystem))
    , mTexture(cs::graphics::TextureLoader::loadFromFile(sTiffFile))
    , mObjectName(std::move(objectName)) {

  // Disables a warning in MSVC about using fopen_s and fscanf_s, which aren't supported in GCC.
  CS_WARNINGS_PUSH
  CS_DISABLE_MSVC_WARNING(4996)

  // load metadata -----------------------------------------------------------
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  FILE* pFile = fopen(sTabFile.c_str(), "r");

  if (pFile == nullptr) {
    logger().error("Failed to add Sharad data: Cannot open file '{}'!", sTabFile);
    return;
  }

  std::vector<ProfileRadarData> meta;

  ProfileRadarData dataElement{};

  // Scan the File, this is specific to the one SHARAD we currently have
  while ( // NOLINTNEXTLINE(cert-err34-c)
      fscanf(pFile, "%d,%d-%d-%dT%d:%d:%d.%d, %f,%f,%f,%f, %f,%f,%f,%f", &dataElement.Number,
          &dataElement.Year, &dataElement.Month, &dataElement.Day, &dataElement.Hour,
          &dataElement.Minute, &dataElement.Second, &dataElement.Millisecond, &dataElement.Latitude,
          &dataElement.Longitude, &dataElement.SurfaceAltitude, &dataElement.MROAltitude,
          &dataElement.c, &dataElement.d, &dataElement.e, &dataElement.f) == 16) {
    meta.push_back(dataElement);
  }

  CS_WARNINGS_POP

  mSamples = static_cast<int>(meta.size());

  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  fclose(pFile);

  // create geometry ---------------------------------------------------------
  struct Vertex {
    glm::vec2 lngLat;
    glm::vec2 tc;
    float     time;
  };

  std::vector<Vertex> vertices(mSamples * 2);

  for (int i = 0; i < mSamples; ++i) {
    double tTime = cs::utils::convert::time::toSpice(
        boost::posix_time::ptime(boost::gregorian::date(meta[i].Year, meta[i].Month, meta[i].Day),
            boost::posix_time::hours(meta[i].Hour) + boost::posix_time::minutes(meta[i].Minute) +
                boost::posix_time::seconds(meta[i].Second) +
                boost::posix_time::milliseconds(meta[i].Millisecond)));

    if (i == 0) {
      mStartTime = tTime;
    }

    glm::vec2 lngLat(
        cs::utils::convert::toRadians(glm::dvec2(meta[i].Longitude, meta[i].Latitude)));

    float x    = 1.F * static_cast<float>(i) / (static_cast<float>(mSamples) - 1.F);
    auto  time = static_cast<float>(tTime - mStartTime);

    vertices[i * 2 + 0].lngLat = lngLat;
    vertices[i * 2 + 0].tc     = glm::vec2(x, 1.F);
    vertices[i * 2 + 0].time   = time;
    vertices[i * 2 + 1].lngLat = lngLat;
    vertices[i * 2 + 1].tc     = glm::vec2(x, 0.F);
    vertices[i * 2 + 1].time   = time;
  }

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
  mVBO.Release();

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0, &mVBO);

  mVAO.EnableAttributeArray(1);
  mVAO.SpecifyAttributeArrayFloat(
      1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), static_cast<GLuint>(offsetof(Vertex, tc)), &mVBO);

  mVAO.EnableAttributeArray(2);
  mVAO.SpecifyAttributeArrayFloat(
      2, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), static_cast<GLuint>(offsetof(Vertex, time)), &mVBO);

  // create sphere shader ----------------------------------------------------
  mShader.InitVertexShaderFromString(VERT);
  mShader.InitFragmentShaderFromString(FRAG);
  mShader.Link();

  mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
  mUniforms.viewportPosition = mShader.GetUniformLocation("uViewportPos");
  mUniforms.sharadTexture    = mShader.GetUniformLocation("uSharadTexture");
  mUniforms.depthBuffer      = mShader.GetUniformLocation("uDepthBuffer");
  mUniforms.sceneScale       = mShader.GetUniformLocation("uSceneScale");
  mUniforms.heightScale      = mShader.GetUniformLocation("uHeightScale");
  mUniforms.radii            = mShader.GetUniformLocation("uRadii");
  mUniforms.time             = mShader.GetUniformLocation("uTime");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double Sharad::getStartTime() const {
  return mStartTime;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Sharad::update(double tTime, double sceneScale) {
  mCurrTime   = tTime;
  mSceneScale = sceneScale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Sharad::Do() {
  auto object = mSolarSystem->getObject(mObjectName);

  if (object && object->getIsBodyVisible()) {
    cs::utils::FrameStats::ScopedTimer timer("Sharad");

    mShader.Bind();

    // get modelview and projection matrices
    std::array<GLfloat, 16> glMatMV{};
    std::array<GLfloat, 16> glMatP{};
    glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
    glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
    auto matMV =
        glm::make_mat4x4(glMatMV.data()) * glm::mat4(object->getObserverRelativeTransform());
    glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(matMV));
    glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());

    std::array<GLint, 4> iViewport{};
    glGetIntegerv(GL_VIEWPORT, iViewport.data());
    mShader.SetUniform(mUniforms.viewportPosition, static_cast<float>(iViewport.at(0)),
        static_cast<float>(iViewport.at(1)));

    auto radii = object->getRadii();

    mShader.SetUniform(mUniforms.sharadTexture, 0);
    mShader.SetUniform(mUniforms.depthBuffer, 1);
    mShader.SetUniform(mUniforms.sceneScale, static_cast<float>(mSceneScale));
    mShader.SetUniform(mUniforms.heightScale, mSettings->mGraphics.pHeightScale.get());
    mShader.SetUniform(mUniforms.radii, static_cast<float>(radii[0]), static_cast<float>(radii[1]),
        static_cast<float>(radii[2]));
    mShader.SetUniform(mUniforms.time, static_cast<float>(mCurrTime - mStartTime));

    mTexture->Bind(GL_TEXTURE0);
    mGraphicsEngine->bindCurrentDepthBufferAsTexture(GL_TEXTURE1, false);

    glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // glDepthFunc(GL_GEQUAL);
    // glDepthMask(false);
    glDisable(GL_DEPTH_TEST);

    // draw --------------------------------------------------------------------
    mVAO.Bind();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, mSamples * 2);
    mVAO.Release();

    // clean up ----------------------------------------------------------------
    glEnable(GL_CULL_FACE);
    mTexture->Unbind(GL_TEXTURE0);

    glPopAttrib();

    mShader.Release();
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Sharad::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::sharad
