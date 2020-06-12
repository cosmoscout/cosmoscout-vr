////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SimpleWMSBody.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>
#include <VistaOGLExt/VistaTexture.h>

#include <curlpp/Infos.hpp>
#include <curlpp/Options.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace csp::simplewmsbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string SimpleWMSBody::SPHERE_VERT = R"(
uniform vec3 uSunDirection;
uniform vec3 uRadii;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

// inputs
layout(location = 0) in vec2 iGridPos;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;
out vec3 vCenter;
out vec2 vLonLat;

const float PI = 3.141592654;

void main()
{
    vTexCoords = vec2(iGridPos.x, 1 - iGridPos.y);
    vLonLat.x = iGridPos.x * 2.0 * PI;
    vLonLat.y = (iGridPos.y - 0.5) * PI;
    vPosition = uRadii * vec3(
        -sin(vLonLat.x) * cos(vLonLat.y),
        -cos(vLonLat.y + PI * 0.5),
        -cos(vLonLat.x) * cos(vLonLat.y)
    );
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

const std::string SimpleWMSBody::SPHERE_FRAG = R"(
uniform vec3 uSunDirection;
uniform sampler2D uBackgroundTexture;
uniform sampler2D uWMSTexture;
uniform sampler2D uSecondWMSTexture;
uniform float uAmbientBrightness;
uniform float uSunIlluminance;
uniform float uFarClip;
uniform float uFade;
uniform bool uUseTexture;
uniform bool uUseSecondTexture;

// inputs
in vec2 vTexCoords;
in vec3 vSunDirection;
in vec3 vPosition;
in vec3 vCenter;
in vec2 vLonLat;

// outputs
layout(location = 0) out vec3 oColor;

vec3 SRGBtoLINEAR(vec3 srgbIn)
{
  vec3 bLess = step(vec3(0.04045), srgbIn);
  return mix(srgbIn / vec3(12.92), pow((srgbIn + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
}

void main()
{
    vec3 backColor = texture(uBackgroundTexture, vTexCoords).rgb;
    oColor = backColor;

    if (uUseTexture) {
      // WMS texture
      vec4 texColor = texture(uWMSTexture, vTexCoords);
      oColor = mix(oColor, texColor.rgb, texColor.a);

      // Fade second texture in.
      if(uUseSecondTexture) {
        vec4 secColorA = texture(uSecondWMSTexture, vTexCoords);
        vec3 secColor = mix(backColor, secColorA.rgb, secColorA.a);
        oColor = mix(secColor, oColor, uFade);
      }
    }

    #ifdef ENABLE_HDR
      oColor = SRGBtoLINEAR(oColor);
    #endif

    oColor = oColor * uSunIlluminance;

    #ifdef ENABLE_LIGHTING
      vec3 normal = normalize(vPosition - vCenter);
      float light = max(dot(normal, uSunDirection), 0.0);
      oColor = mix(oColor * uAmbientBrightness, oColor, light);
    #endif

    gl_FragDepth = length(vPosition) / uFarClip;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

SimpleWMSBody::SimpleWMSBody(std::shared_ptr<cs::core::Settings> const& settings,
    std::shared_ptr<cs::core::SolarSystem>                              solarSystem,
    std::shared_ptr<Plugin::Settings> const&                            pluginSettings,
    std::shared_ptr<cs::core::TimeControl> timeControl, std::string const& sCenterName,
    std::string const& sFrameName, double tStartExistence, double tEndExistence)
    : cs::scene::CelestialBody(sCenterName, sFrameName, tStartExistence, tEndExistence)
    , mSettings(settings)
    , mSolarSystem(solarSystem)
    , mRadii(cs::core::SolarSystem::getRadii(sCenterName))
    , mPluginSettings(pluginSettings)
    , mWMSTexture(new VistaTexture(GL_TEXTURE_2D))
    , mSecondWMSTexture(new VistaTexture(GL_TEXTURE_2D)) {
  pVisibleRadius = mRadii[0];
  mTimeControl   = timeControl;

  // For rendering the sphere, we create a 2D-grid which is warped into a sphere in the vertex
  // shader. The vertex positions are directly used as texture coordinates.
  std::vector<float>    vertices(mGridResolutionX * mGridResolutionY * 2);
  std::vector<unsigned> indices((mGridResolutionX - 1) * (2 + 2 * mGridResolutionY));

  for (uint32_t x = 0; x < mGridResolutionX; ++x) {
    for (uint32_t y = 0; y < mGridResolutionY; ++y) {
      vertices[(x * mGridResolutionY + y) * 2 + 0] = 1.f / (mGridResolutionX - 1) * x;
      vertices[(x * mGridResolutionY + y) * 2 + 1] = 1.f / (mGridResolutionY - 1) * y;
    }
  }

  uint32_t index = 0;

  for (uint32_t x = 0; x < mGridResolutionX - 1; ++x) {
    indices[index++] = x * mGridResolutionY;
    for (uint32_t y = 0; y < mGridResolutionY; ++y) {
      indices[index++] = x * mGridResolutionY + y;
      indices[index++] = (x + 1) * mGridResolutionY + y;
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

SimpleWMSBody::~SimpleWMSBody() {
  mSettings->mGraphics.pEnableLighting.disconnect(mEnableLightingConnection);
  mSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SimpleWMSBody::configure(Plugin::Settings::SimpleWMSBody const& settings) {
  if (mSimpleWMSBodySettings.mTexture != settings.mTexture) {
    mBackgroundTexture = cs::graphics::TextureLoader::loadFromFile(settings.mTexture);
  }
  mGridResolutionX = settings.mGridResolutionX.value_or(200);
  mGridResolutionY = settings.mGridResolutionY.value_or(100);

  mSimpleWMSBodySettings = settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SimpleWMSBody::setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun) {
  mSun = sun;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SimpleWMSBody::getIntersection(
    glm::dvec3 const& rayOrigin, glm::dvec3 const& rayDir, glm::dvec3& pos) const {

  glm::dmat4 transform = glm::inverse(getWorldTransform());

  // Transform ray into planet coordinate system.
  glm::dvec4 origin(rayOrigin, 1.0);
  origin = transform * origin;

  glm::dvec4 direction(rayDir, 0.0);
  direction = transform * direction;
  direction = glm::normalize(direction);

  double b    = glm::dot(origin, direction);
  double c    = glm::dot(origin, origin) - mRadii[0] * mRadii[0];
  double fDet = b * b - c;

  if (fDet < 0.0) {
    return false;
  }

  fDet = std::sqrt(fDet);
  pos  = (origin + direction * (-b - fDet));

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double SimpleWMSBody::getHeight(glm::dvec2 lngLat) const {
  // This is why we call them 'SimpleWMSBodies'.
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 SimpleWMSBody::getRadii() const {
  return mRadii;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SimpleWMSBody::Do() {
  std::lock_guard<std::mutex> guard(mWMSMutex);

  if (!getIsInExistence() || !pVisible.get()) {
    return true;
  }

  cs::utils::FrameTimings::ScopedTimer timer("Simple WMS Bodies");

  if (mActiveWMS && mActiveWMS->mTime.has_value()) {
    boost::posix_time::ptime time =
        cs::utils::convert::time::toPosix(mTimeControl->pSimulationTime.get());

    // Select WMS textures to be downloaded. If no pre-fetch is set, only sellect the texture for
    // the current timestep.
    for (int preFetch = -mActiveWMS->mPrefetchCount.value_or(0);
         preFetch <= mActiveWMS->mPrefetchCount.value_or(0); preFetch++) {
      boost::posix_time::time_duration td = boost::posix_time::seconds(mIntervalDuration);
      time = cs::utils::convert::time::toPosix(mTimeControl->pSimulationTime.get()) + td * preFetch;
      boost::posix_time::time_duration timeSinceStart;
      boost::posix_time::ptime         startTime =
          time - boost::posix_time::microseconds(time.time_of_day().fractional_seconds());
      bool inInterval = utils::timeInIntervals(
          startTime, mTimeIntervals, timeSinceStart, mIntervalDuration, mFormat);

      // TODO: fix interval duration
      // mIntervalDuration contains seconds, which are only approx. for months and years.
      // It handles only 1 day/ 1 month/ 1 year intervals for now (with formatting).
      /*if (mIntervalDuration != 0) {
        startTime -= boost::posix_time::seconds(timeSinceStart.total_seconds() % mIntervalDuration);
      }*/
      std::string timeString = utils::timeToString(mFormat.c_str(), startTime);

      // Select a WMS texture over the period of timeDuration if timespan is enabled.
      if (mPluginSettings->mEnableTimespan.get() && mIntervalDuration != 0) {
        boost::posix_time::time_duration timeDuration =
            boost::posix_time::seconds(mIntervalDuration);
        boost::posix_time::ptime intervalAfter = getStartTime(startTime + timeDuration);
        timeString += "/" + utils::timeToString(mFormat.c_str(), intervalAfter);
      }

      auto texture1 = mTextureFilesBuffer.find(timeString);
      auto texture2 = mTexturesBuffer.find(timeString);
      auto texture3 = mTextures.find(timeString);
      auto texture4 = std::find(mWrongTextures.begin(), mWrongTextures.end(), timeString);

      // Only load textures those aren't stored yet.
      if (texture1 == mTextureFilesBuffer.end() && texture2 == mTexturesBuffer.end() &&
          texture3 == mTextures.end() && texture4 == mWrongTextures.end() && inInterval) {
        // Load WMS texture to the disk.
        mTextureFilesBuffer.insert(std::pair<std::string, std::future<std::string>>(
            timeString, mTextureLoader.loadTextureAsync(timeString, mRequest, mActiveWMS->mLayers,
                            mPluginSettings->mMapCache.get())));
      }
    }

    // Check whether the WMS textures are loaded to the disk.
    auto fileIt = mTextureFilesBuffer.begin();
    while (fileIt != mTextureFilesBuffer.end()) {
      if (fileIt->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        std::string fileName = fileIt->second.get();

        if (fileName != "Error") {
          // Load WMS texture to memory
          mTexturesBuffer.insert(std::pair<std::string, std::future<unsigned char*>>(
              fileIt->first, mTextureLoader.loadTextureFromFileAsync(fileName)));
        } else {
          mWrongTextures.emplace_back(fileIt->first);
        }

        fileIt = mTextureFilesBuffer.erase(fileIt);
      } else {
        ++fileIt;
      }
    }

    // Check whether the WMS textures are loaded to the memory.
    auto texIt = mTexturesBuffer.begin();
    while (texIt != mTexturesBuffer.end()) {
      if (texIt->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        mTextures.insert(std::pair<std::string, unsigned char*>(texIt->first, texIt->second.get()));
        texIt = mTexturesBuffer.erase(texIt);
      } else {
        ++texIt;
      }
    }

    time = cs::utils::convert::time::toPosix(mTimeControl->pSimulationTime.get());
    boost::posix_time::time_duration timeSinceStart;
    boost::posix_time::ptime         startTime =
        time - boost::posix_time::microseconds(time.time_of_day().fractional_seconds());
    bool inInterval = utils::timeInIntervals(
        startTime, mTimeIntervals, timeSinceStart, mIntervalDuration, mFormat);

    // TODO: fix interval duration
    /*if (mIntervalDuration != 0) {
      startTime -= boost::posix_time::seconds(timeSinceStart.total_seconds() % mIntervalDuration);
    }*/

    boost::posix_time::time_duration timeDuration = boost::posix_time::seconds(mIntervalDuration);
    std::string                      timeString   = utils::timeToString(mFormat.c_str(), startTime);

    if (mPluginSettings->mEnableTimespan.get() && mIntervalDuration != 0) {
      boost::posix_time::ptime intervalAfter = getStartTime(startTime + timeDuration);
      timeString += "/" + utils::timeToString(mFormat.c_str(), intervalAfter);
    }
    auto tex = mTextures.find(timeString);

    // Use Wms texture inside the interval.
    if (inInterval && tex != mTextures.end()) {
      // Only update if we have a new texture.
      if (mCurrentTexture != timeString) {
        mWMSTextureUsed = true;
        mWMSTexture->UploadTexture(mActiveWMS->mWidth, mActiveWMS->mHeight, tex->second, false);
        mCurrentTexture = timeString;
      }
    } // Use default planet texture instead.
    else {
      mWMSTextureUsed = false;
      mCurrentTexture = "";
    }

    if (!mWMSTextureUsed || !mPluginSettings->mEnableInterpolation.get() ||
        mIntervalDuration == 0) {
      mSecondWMSTextureUsed = false;
      mCurrentSecondTexture = "";
    } // Create fading between Wms textures when interpolation is enabled.
    else {
      boost::posix_time::ptime intervalAfter = getStartTime(startTime + timeDuration);
      tex = mTextures.find(utils::timeToString(mFormat.c_str(), intervalAfter));

      if (tex != mTextures.end()) {
        // Only update if we ha a new second texture.
        if (mCurrentSecondTexture != utils::timeToString(mFormat.c_str(), intervalAfter)) {
          mSecondWMSTexture->UploadTexture(
              mActiveWMS->mWidth, mActiveWMS->mHeight, tex->second, false);
          mCurrentSecondTexture = utils::timeToString(mFormat.c_str(), intervalAfter);
          mSecondWMSTextureUsed = true;
        }
        // Interpolate fade value between the 2 WMS textures.
        mFade = static_cast<float>((double)(intervalAfter - time).total_seconds() /
                                   (double)(intervalAfter - startTime).total_seconds());
      }
    }
  }

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

    mShaderDirty = false;
  }

  mShader.Bind();

  glm::vec3 sunDirection(1, 0, 0);
  float     sunIlluminance(1.f);
  float     ambientBrightness(mSettings->mGraphics.pAmbientBrightness.get());

  if (getCenterName() == "Sun") {
    // If the SimpleWMSBody is actually the sun, we have to calculate the lighting differently.
    if (mSettings->mGraphics.pEnableHDR.get()) {
      double sceneScale = 1.0 / mSolarSystem->getObserver().getAnchorScale();
      sunIlluminance    = static_cast<float>(
          mSolarSystem->pSunLuminousPower.get() /
          (sceneScale * sceneScale * mRadii[0] * mRadii[0] * 4.0 * glm::pi<double>()));
    }

    ambientBrightness = 1.0f;

  } else if (mSun) {
    // For all other bodies we can use the utility methods from the SolarSystem.
    if (mSettings->mGraphics.pEnableHDR.get()) {
      sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(getWorldTransform()[3]));
    }

    sunDirection = mSolarSystem->getSunDirection(getWorldTransform()[3]);
  }

  // Set uniforms.
  mShader.SetUniform(mShader.GetUniformLocation("uSunDirection"), sunDirection[0], sunDirection[1],
      sunDirection[2]);
  mShader.SetUniform(mShader.GetUniformLocation("uSunIlluminance"), sunIlluminance);
  mShader.SetUniform(mShader.GetUniformLocation("uAmbientBrightness"), ambientBrightness);
  mShader.SetUniform(mShader.GetUniformLocation("uUseTexture"), mWMSTextureUsed);
  mShader.SetUniform(mShader.GetUniformLocation("uUseSecondTexture"), mSecondWMSTextureUsed);

  // Get modelview and projection matrices.
  GLfloat glMatMV[16], glMatP[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, &glMatMV[0]);
  glGetFloatv(GL_PROJECTION_MATRIX, &glMatP[0]);
  auto matMV = glm::make_mat4x4(glMatMV) * glm::mat4(getWorldTransform());
  glUniformMatrix4fv(
      mShader.GetUniformLocation("uMatModelView"), 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mShader.GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP);

  mShader.SetUniform(mShader.GetUniformLocation("uBackgroundTexture"), 0);
  mShader.SetUniform(mShader.GetUniformLocation("uWMSTexture"), 1);
  mShader.SetUniform(mShader.GetUniformLocation("uSecondWMSTexture"), 2);
  mShader.SetUniform(
      mShader.GetUniformLocation("uRadii"), (float)mRadii[0], (float)mRadii[0], (float)mRadii[0]);
  mShader.SetUniform(
      mShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());

  // Only bind the enabled textures.
  mBackgroundTexture->Bind(GL_TEXTURE0);
  if (mWMSTextureUsed) {
    mWMSTexture->Bind(GL_TEXTURE1);

    if (mSecondWMSTextureUsed) {
      mShader.SetUniform(mShader.GetUniformLocation("uFade"), mFade);
      mSecondWMSTexture->Bind(GL_TEXTURE2);
    }
  }

  // Draw.
  mSphereVAO.Bind();
  glDrawElements(GL_TRIANGLE_STRIP, (mGridResolutionX - 1) * (2 + 2 * mGridResolutionY),
      GL_UNSIGNED_INT, nullptr);
  mSphereVAO.Release();

  // Clean up.
  mBackgroundTexture->Unbind(GL_TEXTURE0);

  if (mWMSTextureUsed) {
    mWMSTexture->Unbind(GL_TEXTURE1);

    if (mSecondWMSTextureUsed) {
      mSecondWMSTexture->Unbind(GL_TEXTURE2);
    }
  }

  mShader.Release();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SimpleWMSBody::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

boost::posix_time::ptime SimpleWMSBody::getStartTime(boost::posix_time::ptime time) {
  boost::posix_time::time_duration timeSinceStart;
  boost::posix_time::ptime         startTime;

  bool inInterval =
      utils::timeInIntervals(time, mTimeIntervals, timeSinceStart, mIntervalDuration, mFormat);

  if (inInterval) {
    /*startTime =
        time - boost::posix_time::seconds(timeSinceStart.total_seconds() % mIntervalDuration);*/
    startTime = time;	// TODO: fix interval duration
  } else {
    startTime = time;
  }

  return startTime;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SimpleWMSBody::setActiveWMS(std::shared_ptr<Plugin::Settings::WMSConfig> wms) {
  mTextures.clear();
  mTexturesBuffer.clear();
  mTextureFilesBuffer.clear();
  mWrongTextures.clear();
  mTimeIntervals.clear();
  mWMSTextureUsed       = false;
  mSecondWMSTextureUsed = false;
  mCurrentTexture       = "";
  mCurrentSecondTexture = "";
  mActiveWMS            = wms;

  if (mActiveWMS) {
    // Create request URL for map server.
    std::stringstream url;
    url << mActiveWMS->mUrl << "&WIDTH=" << mActiveWMS->mWidth << "&HEIGHT=" << mActiveWMS->mHeight
        << "&LAYERS=" << mActiveWMS->mLayers;
    mRequest = url.str();

    // Set time intervals and format if it is defined in config.
    if (mActiveWMS->mTime.has_value()) {
      utils::parseIsoString(mActiveWMS->mTime.value(), mTimeIntervals);
      mIntervalDuration = mTimeIntervals.at(0).mIntervalDuration;
      mFormat           = mTimeIntervals.at(0).mFormat;
    } // Download WMS texture without timestep.
    else {
      std::string cacheFile = mTextureLoader.loadTexture(
          "", mRequest, mActiveWMS->mLayers, mPluginSettings->mMapCache.get());
      if (cacheFile != "Error") {
        mWMSTexture     = cs::graphics::TextureLoader::loadFromFile(cacheFile);
        mWMSTextureUsed = true;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<TimeInterval> SimpleWMSBody::getTimeIntervals() {
  return mTimeIntervals;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::simplewmsbodies
