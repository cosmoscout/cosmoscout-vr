// Plugin Includes
#include "TextureOverlayRenderer.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "../../../src/cs-utils/utils.hpp"

// VISTA includes
#include <VistaInterProcComm/Connections/VistaByteBufferDeSerializer.h>
#include <VistaKernel/Cluster/VistaClusterMode.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaOGLExt/Rendering/ABuffer/VistaABufferOIT.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

// Standard includes
#include <boost/filesystem.hpp>
#include <functional>
#include <glm/gtc/type_ptr.hpp>
#include <json.hpp>

#include <cmath>

#define _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING
using json = nlohmann::json;

namespace csp::wmsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr float PI = 3.141592654f;

////////////////////////////////////////////////////////////////////////////////////////////////////

TextureOverlayRenderer::TextureOverlayRenderer(std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::TimeControl>                                            timeControl,
    std::shared_ptr<Plugin::Settings> const& pluginSettings)
    : mSolarSystem(solarSystem)
    , mTimeControl(timeControl)
    , mPluginSettings(pluginSettings)
    , mWMSTexture(new VistaTexture(GL_TEXTURE_2D))
    , mSecondWMSTexture(new VistaTexture(GL_TEXTURE_2D))
    , mTexture(new VistaTexture(GL_TEXTURE_2D)) {
  logger().debug("[TextureOverlayRenderer] Compiling shader");

  m_pSurfaceShader = nullptr;
  m_pSurfaceShader = new VistaGLSLShader();
  m_pSurfaceShader->InitVertexShaderFromString(SURFACE_VERT);
  m_pSurfaceShader->InitFragmentShaderFromString(SURFACE_FRAG);
  m_pSurfaceShader->InitGeometryShaderFromString(SURFACE_GEOM);
  m_pSurfaceShader->Link();

  // create textures ---------------------------------------------------------
  for (auto const& viewport : GetVistaSystem()->GetDisplayManager()->GetViewports()) {
    GBufferData bufferData;

    // Texture for previous renderer depth buffer
    bufferData.mDepthBuffer = new VistaTexture(GL_TEXTURE_RECTANGLE);
    bufferData.mDepthBuffer->Bind();
    bufferData.mDepthBuffer->SetWrapS(GL_CLAMP);
    bufferData.mDepthBuffer->SetWrapT(GL_CLAMP);
    bufferData.mDepthBuffer->SetMinFilter(GL_NEAREST);
    bufferData.mDepthBuffer->SetMagFilter(GL_NEAREST);
    bufferData.mDepthBuffer->Unbind();

    // Color texture to overlay
    bufferData.mColorBuffer = new VistaTexture(GL_TEXTURE_2D);
    bufferData.mColorBuffer->Bind();
    bufferData.mColorBuffer->SetWrapS(GL_CLAMP);
    bufferData.mColorBuffer->SetWrapT(GL_CLAMP);
    bufferData.mColorBuffer->SetMinFilter(GL_LINEAR);
    bufferData.mColorBuffer->SetMagFilter(GL_LINEAR);
    bufferData.mColorBuffer->Unbind();

    mGBufferData[viewport.second] = bufferData;
  }
  logger().debug("[TextureOverlayRenderer] Compiling shader done");

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::ePlanets) + 10);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TextureOverlayRenderer::~TextureOverlayRenderer() {
  for (auto data : mGBufferData) {
    delete data.second.mDepthBuffer;
    delete data.second.mColorBuffer;
  }

  clearTextures();

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::configure(Plugin::Settings::SimpleWMSBody const& settings) {
  mSimpleWMSBodySettings = settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::setActiveWMS(std::shared_ptr<Plugin::Settings::WMSConfig> wms) {
  clearTextures();
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
      mSampleDuration = mTimeIntervals.at(0).mSampleDuration;
      mFormat         = mTimeIntervals.at(0).mFormat;
    } // Download WMS texture without timestep.
    else {
      std::string cacheFile = mTextureLoader.loadTexture("", mRequest, mActiveWMS->mFormat,
          mActiveWMS->mLayers, mActiveWMS->mLatRange.get(), mActiveWMS->mLonRange.get(),
          mPluginSettings->mMapCache.get());
      if (cacheFile != "Error") {
        mWMSTexture     = cs::graphics::TextureLoader::loadFromFile(cacheFile);
        mWMSTextureUsed = true;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::clearTextures() {
  auto texIt = mTextures.begin();
  while (texIt != mTextures.end()) {
    delete texIt->second;
    texIt++;
  }

  mTextures.clear();
  mTexturesBuffer.clear();
  mTextureFilesBuffer.clear();
  mWrongTextures.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::SetOverlayTexture(std::shared_ptr<VistaTexture> texture) {
  mTexture = texture;
  mTexture->Bind();
  mTexture->SetWrapS(GL_CLAMP_TO_EDGE);
  mTexture->SetWrapT(GL_CLAMP_TO_EDGE);
  mTexture->Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::SetBounds(std::array<double, 4> bounds) {
  mLngLatBounds = bounds;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<TimeInterval> TextureOverlayRenderer::getTimeIntervals() {
  return mTimeIntervals;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TextureOverlayRenderer::Do() {
  if (mActiveWMS && mActiveWMS->mTime.has_value()) {
    // Get the current time. Pre-fetch times are related to this.
    boost::posix_time::ptime time =
        cs::utils::convert::time::toPosix(mTimeControl->pSimulationTime.get());

    // Select WMS textures to be downloaded. If no pre-fetch is set, only sellect the texture for
    // the current timestep.
    for (int preFetch = -mActiveWMS->mPrefetchCount.value_or(0);
         preFetch <= mActiveWMS->mPrefetchCount.value_or(0); preFetch++) {

      // Get the start time of the WMS sample.
      boost::posix_time::ptime sampleStartTime =
          utils::addDurationToTime(time, mSampleDuration, preFetch);
      sampleStartTime -= boost::posix_time::microseconds(time.time_of_day().fractional_seconds());
      bool inInterval =
          utils::timeInIntervals(sampleStartTime, mTimeIntervals, mSampleDuration, mFormat);

      // Create identifier for the sample start time.
      std::string timeString = utils::timeToString(mFormat.c_str(), sampleStartTime);

      // Select a WMS texture over the period of timeDuration if timespan is enabled.
      if (mPluginSettings->mEnableTimespan.get() && mSampleDuration.isDuration()) {
        boost::posix_time::ptime sampleAfter =
            utils::addDurationToTime(sampleStartTime, mSampleDuration);
        bool isAfterInInterval =
            utils::timeInIntervals(sampleAfter, mTimeIntervals, mSampleDuration, mFormat);

        // Select timespan only when the sample after is also in the intervals.
        if (isAfterInInterval) {
          timeString += "/" + utils::timeToString(mFormat.c_str(), sampleAfter);
        }
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
            timeString, mTextureLoader.loadTextureAsync(timeString, mRequest, mActiveWMS->mFormat,
                            mActiveWMS->mLayers, mActiveWMS->mLatRange.get(),
                            mActiveWMS->mLonRange.get(), mPluginSettings->mMapCache.get())));
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
        unsigned char* texture = texIt->second.get();

        if (strcmp(reinterpret_cast<const char*>(texture), const_cast<char*>("Error")) != 0) {
          mTextures.insert(std::pair<std::string, unsigned char*>(texIt->first, texture));
        } else {
          mWrongTextures.emplace_back(texIt->first);
        }

        texIt = mTexturesBuffer.erase(texIt);
      } else {
        ++texIt;
      }
    }

    // Get the current time.
    time = cs::utils::convert::time::toPosix(mTimeControl->pSimulationTime.get());
    boost::posix_time::ptime sampleStartTime =
        time - boost::posix_time::microseconds(time.time_of_day().fractional_seconds());
    bool inInterval =
        utils::timeInIntervals(sampleStartTime, mTimeIntervals, mSampleDuration, mFormat);

    // Create identifier for the sample start time.
    std::string timeString = utils::timeToString(mFormat.c_str(), sampleStartTime);

    // Select a WMS texture over the period of timeDuration if timespan is enabled.
    if (mPluginSettings->mEnableTimespan.get() && mSampleDuration.isDuration()) {
      boost::posix_time::ptime sampleAfter =
          utils::addDurationToTime(sampleStartTime, mSampleDuration);
      bool isAfterInInterval =
          utils::timeInIntervals(sampleAfter, mTimeIntervals, mSampleDuration, mFormat);

      // Select timespan only when the sample after is also in the intervals.
      if (isAfterInInterval) {
        timeString += "/" + utils::timeToString(mFormat.c_str(), sampleAfter);
      }
    }

    // Find the current texture.
    auto tex = mTextures.find(timeString);

    // Use Wms texture inside the interval.
    if (inInterval && tex != mTextures.end()) {
      // Only update if we have a new texture.
      if (mCurrentTexture != timeString) {
        mWMSTextureUsed = true;
        mWMSTexture->UploadTexture(mActiveWMS->mWidth, mActiveWMS->mHeight, tex->second, false);
        mCurrentTexture = timeString;

        SetOverlayTexture(mWMSTexture);
        std::array<double, 4> bounds;
        bounds[0] = mActiveWMS->mLonRange.get()[0] / 180. * PI;
        bounds[1] = mActiveWMS->mLatRange.get()[1] / 180. * PI;
        bounds[2] = mActiveWMS->mLonRange.get()[1] / 180. * PI;
        bounds[3] = mActiveWMS->mLatRange.get()[0] / 180. * PI;
        SetBounds(bounds);
      }
    } // Use default planet texture instead.
    else {
      mWMSTextureUsed = false;
      mCurrentTexture = "";
    }

    if (!mWMSTextureUsed || !mPluginSettings->mEnableInterpolation.get() ||
        !mSampleDuration.isDuration()) {
      mSecondWMSTextureUsed = false;
      mCurrentSecondTexture = "";
    } // Create fading between Wms textures when interpolation is enabled.
    else {
      boost::posix_time::ptime sampleAfter =
          utils::addDurationToTime(sampleStartTime, mSampleDuration);
      bool isAfterInInterval =
          utils::timeInIntervals(sampleAfter, mTimeIntervals, mSampleDuration, mFormat);

      // Find texture for the following sample.
      tex = mTextures.find(utils::timeToString(mFormat.c_str(), sampleAfter));

      if (isAfterInInterval && tex != mTextures.end()) {
        // Only update if we ha a new second texture.
        if (mCurrentSecondTexture != utils::timeToString(mFormat.c_str(), sampleAfter)) {
          mSecondWMSTexture->UploadTexture(
              mActiveWMS->mWidth, mActiveWMS->mHeight, tex->second, false);
          mCurrentSecondTexture = utils::timeToString(mFormat.c_str(), sampleAfter);
          mSecondWMSTextureUsed = true;
        }
        // Interpolate fade value between the 2 WMS textures.
        mFade = static_cast<float>((double)(sampleAfter - time).total_seconds() /
                                   (double)(sampleAfter - sampleStartTime).total_seconds());
      }
    }
  }

  // get active planet
  if (mSolarSystem->pActiveBody.get() == nullptr ||
      mSolarSystem->pActiveBody.get()->getCenterName() != "Earth") {
    return false;
  }

  // save current lighting and meterial state of the OpenGL state machine
  glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT);
  glEnable(GL_TEXTURE_2D);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  glEnable(GL_BLEND);

  double nearClip = NAN;
  double farClip  = NAN;
  GetVistaSystem()
      ->GetDisplayManager()
      ->GetCurrentRenderInfo()
      ->m_pViewport->GetProjection()
      ->GetProjectionProperties()
      ->GetClippingRange(nearClip, farClip);

  // copy depth buffer from previous rendering
  // -------------------------------------------------------
  GLint iViewport[4];
  glGetIntegerv(GL_VIEWPORT, iViewport);

  auto*       viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  auto const& data     = mGBufferData[viewport];

  data.mDepthBuffer->Bind();
  glCopyTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT, iViewport[0], iViewport[1],
      iViewport[2], iViewport[3], 0);

  // get matrices and related values -----------------------------------------
  GLfloat glMatP[16];
  GLfloat glMatMV[16];
  glGetFloatv(GL_PROJECTION_MATRIX, &glMatP[0]);
  glGetFloatv(GL_MODELVIEW_MATRIX, &glMatMV[0]);

  std::string closestPlanet     = mSolarSystem->pActiveBody.get()->getCenterName();
  auto        activeBody        = mSolarSystem->pActiveBody.get();
  glm::dmat4  matWorldTransform = activeBody->getWorldTransform();

  VistaTransformMatrix matM(glm::value_ptr(matWorldTransform), true);
  VistaTransformMatrix matMV(matM);
  VistaTransformMatrix matInvMV(matMV.GetInverted());
  VistaTransformMatrix matInvP(VistaTransformMatrix(glMatP, true).GetInverted());
  VistaTransformMatrix matInvMVP(matInvMV * matInvP);
  // get matrices and related values -----------------------------------------

  // Bind shader before draw
  m_pSurfaceShader->Bind();

  data.mDepthBuffer->Bind(GL_TEXTURE0);
  mTexture->Bind(GL_TEXTURE1);

  m_pSurfaceShader->SetUniform(m_pSurfaceShader->GetUniformLocation("uDepthBuffer"), 0);
  m_pSurfaceShader->SetUniform(m_pSurfaceShader->GetUniformLocation("uTexture"), 1);

  // Why is there no set uniform for matrices??? //TODO: There is one
  glm::dmat4 InverseWorldTransform = glm::inverse(matWorldTransform);
  GLint      loc                   = m_pSurfaceShader->GetUniformLocation("uMatInvMV");
  // glUniformMatrix4fv(loc, 1, GL_FALSE, matInvMV.GetData());
  glUniformMatrix4dv(loc, 1, GL_FALSE, glm::value_ptr(InverseWorldTransform));
  loc = m_pSurfaceShader->GetUniformLocation("uMatInvMVP");
  glUniformMatrix4fv(loc, 1, GL_FALSE, matInvMVP.GetData());
  loc = m_pSurfaceShader->GetUniformLocation("uMatInvP");
  glUniformMatrix4fv(loc, 1, GL_FALSE, matInvP.GetData());
  loc = m_pSurfaceShader->GetUniformLocation("uMatMV");
  glUniformMatrix4fv(loc, 1, GL_FALSE, matMV.GetData());

  m_pSurfaceShader->SetUniform(
      m_pSurfaceShader->GetUniformLocation("uFarClip"), static_cast<float>(farClip));

  // Double precision bounds
  loc = m_pSurfaceShader->GetUniformLocation("uBounds");
  glUniform4dv(loc, 1, mLngLatBounds.data());

  glm::vec4 sunDirection =
      glm::normalize(glm::inverse(matWorldTransform) *
                     (mSolarSystem->getSun()->getWorldTransform()[3] - matWorldTransform[3]));
  m_pSurfaceShader->SetUniform(m_pSurfaceShader->GetUniformLocation("uSunDirection"),
      sunDirection[0], sunDirection[1], sunDirection[2]);

  // provide radii to shader
  auto mRadii = cs::core::SolarSystem::getRadii(mSolarSystem->pActiveBody.get()->getCenterName());
  m_pSurfaceShader->SetUniform(m_pSurfaceShader->GetUniformLocation("uRadii"),
      static_cast<float>(mRadii[0]), static_cast<float>(mRadii[1]), static_cast<float>(mRadii[2]));

  int depthBits = 0;
  glGetIntegerv(GL_DEPTH_BITS, &depthBits);

  // Dummy draw
  glDrawArrays(GL_POINTS, 0, 1);

  data.mDepthBuffer->Unbind(GL_TEXTURE0);
  data.mColorBuffer->Unbind(GL_TEXTURE1);

  // Release shader
  m_pSurfaceShader->Release();

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glPopAttrib();
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TextureOverlayRenderer::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float fMin[3] = {-6371000.0f, -6371000.0f, -6371000.0f};
  float fMax[3] = {6371000.0f, 6371000.0f, 6371000.0f};

  oBoundingBox.SetBounds(fMin, fMax);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
