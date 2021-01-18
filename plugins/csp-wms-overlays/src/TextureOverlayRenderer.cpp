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

TextureOverlayRenderer::TextureOverlayRenderer(std::string center,
    std::shared_ptr<cs::core::SolarSystem>                 solarSystem,
    std::shared_ptr<cs::core::TimeControl>                 timeControl,
    std::shared_ptr<Plugin::Settings> const&               pluginSettings)
    : mCenterName(center)
    , mSolarSystem(solarSystem)
    , mTimeControl(timeControl)
    , mPluginSettings(pluginSettings)
    , mMaxSize(pluginSettings->mMaxTextureSize.get())
    , mWMSTexture(new VistaTexture(GL_TEXTURE_2D))
    , mSecondWMSTexture(new VistaTexture(GL_TEXTURE_2D))
    , mMinBounds({(float)-solarSystem->getRadii(center)[0],
          (float)-solarSystem->getRadii(center)[1], (float)-solarSystem->getRadii(center)[2]})
    , mMaxBounds({(float)solarSystem->getRadii(center)[0], (float)solarSystem->getRadii(center)[1],
          (float)solarSystem->getRadii(center)[2]}) {
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
  mWMSTexture->Bind();
  mWMSTexture->SetWrapS(GL_CLAMP_TO_EDGE);
  mWMSTexture->SetWrapT(GL_CLAMP_TO_EDGE);
  mWMSTexture->Unbind();

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

std::string TextureOverlayRenderer::getCenter() const {
  return mCenterName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::configure(Plugin::Settings::Body const& settings) {
  mSimpleWMSOverlaySettings = settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::setActiveWMS(
    std::shared_ptr<WebMapService> wms, std::shared_ptr<WebMapLayer> layer) {
  clearTextures();
  mTimeIntervals.clear();

  mWMSTextureUsed       = false;
  mSecondWMSTextureUsed = false;
  mCurrentTexture       = "";
  mCurrentSecondTexture = "";
  mActiveWMS            = wms;
  mActiveWMSLayer       = layer;
  if (mActiveWMSLayer && mActiveWMSLayer->isRequestable()) {
    mLonRange = mActiveWMSLayer->getSettings().mLonRange;
    mLatRange = mActiveWMSLayer->getSettings().mLatRange;

    if (mActiveWMSLayer->getSettings().mTime.has_value()) {
      utils::parseIsoString(mActiveWMSLayer->getSettings().mTime.value(), mTimeIntervals);
      mSampleDuration = mTimeIntervals.at(0).mSampleDuration;
      mFormat         = mTimeIntervals.at(0).mFormat;
    } else {
      // Download WMS texture without timestep.
      WebMapTextureLoader::Request request;
      request.mMaxSize = mMaxSize;
      request.mStyle   = mStyle;

      std::optional<WebMapTextureFile> cacheFile = mTextureLoader.loadTexture(
          *mActiveWMS, *mActiveWMSLayer, request, mPluginSettings->mMapCache.get());
      if (cacheFile.has_value()) {
        mLonRange = cacheFile->mLonRange;
        mLatRange = cacheFile->mLatRange;

        mWMSTexture = cs::graphics::TextureLoader::loadFromFile(cacheFile->mPath);
        mWMSTexture->Bind();
        mWMSTexture->SetWrapS(GL_CLAMP_TO_EDGE);
        mWMSTexture->SetWrapT(GL_CLAMP_TO_EDGE);
        mWMSTexture->Unbind();
        mWMSTextureUsed = true;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::setStyle(std::string style) {
  mStyle = style;
  clearTextures();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::clearTextures() {
  auto texIt = mTextures.begin();
  while (texIt != mTextures.end()) {
    delete texIt->second.mData;
    texIt++;
  }

  mTextures.clear();
  mTexturesBuffer.clear();
  mTextureFilesBuffer.clear();
  mWrongTextures.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::updateLonLatRange() {
  if (!mActiveWMS || !mActiveWMSLayer) {
    return;
  }

  clearTextures();

  VistaProjection::VistaProjectionProperties* projectionProperties =
      GetVistaSystem()
          ->GetDisplayManager()
          ->GetCurrentRenderInfo()
          ->m_pViewport->GetProjection()
          ->GetProjectionProperties();

  float posX, posY, posZ;
  projectionProperties->GetProjPlaneMidpoint(posX, posY, posZ);
  double near, far;
  projectionProperties->GetClippingRange(near, far);
  double left, right, bottom, top;
  projectionProperties->GetProjPlaneExtents(left, right, bottom, top);

  // Get the intersections of the camera rays at the corners of the screen with the body.
  std::array<std::pair<bool, glm::dvec3>, 4> intersections;
  intersections[0].first =
      mSolarSystem->getBody(mCenterName)
          ->getIntersection(glm::dvec3(0, 0, 0), glm::normalize(glm::dvec3(left, top, posZ)),
              intersections[0].second);
  intersections[1].first =
      mSolarSystem->getBody(mCenterName)
          ->getIntersection(glm::dvec3(0, 0, 0), glm::normalize(glm::dvec3(left, bottom, posZ)),
              intersections[1].second);
  intersections[2].first =
      mSolarSystem->getBody(mCenterName)
          ->getIntersection(glm::dvec3(0, 0, 0), glm::normalize(glm::dvec3(right, bottom, posZ)),
              intersections[2].second);
  intersections[3].first =
      mSolarSystem->getBody(mCenterName)
          ->getIntersection(glm::dvec3(0, 0, 0), glm::normalize(glm::dvec3(right, top, posZ)),
              intersections[3].second);

  if (!std::all_of(intersections.begin(), intersections.end(),
          [](auto intersection) { return intersection.first; })) {
    // The body is not visible in all four corners of the screen.
    // For now this results use the maximum bounds of the map.
    mLonRange = mActiveWMSLayer->getSettings().mLonRange;
    mLatRange = mActiveWMSLayer->getSettings().mLatRange;
  } else {
    // All four corners of the screen show the body.
    // The intersection points can be converted to longitude and latitude.
    glm::dvec3                radii = mSolarSystem->getRadii(mCenterName);
    std::array<glm::dvec2, 4> screenBounds;
    for (int i = 0; i < 4; i++) {
      screenBounds[i] = cs::utils::convert::cartesianToLngLat(intersections[i].second, radii);
      screenBounds[i] = cs::utils::convert::toDegrees(screenBounds[i]);
    }

    mLonRange[0] = screenBounds[0][0];
    mLonRange[1] = screenBounds[0][0];

    // Determine the minimum and maximum longitude.
    // To do so, the edges between neighboring corners are examined and classified as one of four
    // categories. Depending on the category the longitude range can be updated.
    // Also save the lengths of the edges for later (lonDiffs).
    // Uses counterclockwise winding order.
    std::array<double, 4> lonDiffs;
    double                offset = 0;
    for (int i = 1; i < 5; i++) {
      if (screenBounds[i % 4][0] > screenBounds[i - 1][0]) {
        if (screenBounds[i % 4][0] - screenBounds[i - 1][0] < 180) {
          // 0  90  180 270 360
          // | x---x |   |   |
          //   1   2
          // West to east, dateline is not crossed
          mLonRange[1]    = std::max(mLonRange[1], screenBounds[i % 4][0] + offset);
          lonDiffs[i - 1] = screenBounds[i % 4][0] - screenBounds[i - 1][0];
        } else {
          // 0  90  180 270 360
          // --x |   |   | x--
          //   1           2
          // East to west, dateline is crossed
          mLonRange[0]    = std::min(mLonRange[0] + 360, screenBounds[i % 4][0]);
          mLonRange[1]    = mLonRange[1] + 360;
          lonDiffs[i - 1] = screenBounds[i % 4][0] - (screenBounds[i - 1][0] + 360);
        }
      } else {
        if (screenBounds[i - 1][0] - screenBounds[i % 4][0] < 180) {
          // 0  90  180 270 360
          // | x---x |   |   |
          //   2   1
          // East to west, dateline is not crossed
          mLonRange[0]    = std::min(mLonRange[0], screenBounds[i % 4][0] + offset);
          lonDiffs[i - 1] = screenBounds[i % 4][0] - screenBounds[i - 1][0];
        } else {
          // 0  90  180 270 360
          // --x |   |   | x--
          //   2           1
          // West to East, dateline is crossed
          mLonRange[1]    = std::max(mLonRange[1], screenBounds[i % 4][0] + 360);
          offset          = 360;
          lonDiffs[i - 1] = (screenBounds[i % 4][0] + 360) - screenBounds[i - 1][0];
        }
      }
    }
    if (mLonRange[1] > 360) {
      mLonRange[0] -= 360;
      mLonRange[1] -= 360;
    }

    std::array<double, 4> lats;
    std::transform(screenBounds.begin(), screenBounds.end(), lats.begin(),
        [](glm::dvec2 corner) { return corner[1]; });

    mLatRange[0] = *std::min_element(lats.begin(), lats.end());
    mLatRange[1] = *std::max_element(lats.begin(), lats.end());

    // Check if the longitude range spans the whole earth, which would mean that one of the poles is
    // visible. >= 270 is used instead of >= 360 to prevent floating point errors.
    // As long as no pole is visible the maximum range should be 180 degrees, so this check can not
    // result in false positives.
    if (mLonRange[1] - mLonRange[0] >= 270) {
      // 360 degree ranges other than [-180, 180] result in problems on some servers.
      mLonRange = {-180, 180};
      if (std::all_of(lonDiffs.begin(), lonDiffs.end(), [](double diff) { return diff > 0; })) {
        // West to east => north pole is visible
        mLatRange[1] = 90;
      } else if (std::all_of(
                     lonDiffs.begin(), lonDiffs.end(), [](double diff) { return diff < 0; })) {
        // East to west => south pole is visible
        mLatRange[0] = -90;
      } else {
        logger().warn("Could not determine which pole is visible");
      }
    }
  }

  if (!mActiveWMSLayer->getSettings().mTime.has_value()) {
    WebMapTextureLoader::Request request;
    request.mMaxSize  = mMaxSize;
    request.mStyle    = mStyle;
    request.mLonRange = mLonRange;
    request.mLatRange = mLatRange;

    std::optional<WebMapTextureFile> cacheFile = mTextureLoader.loadTexture(
        *mActiveWMS, *mActiveWMSLayer, request, mPluginSettings->mMapCache.get());
    if (cacheFile.has_value()) {
      mLonRange = cacheFile->mLonRange;
      mLatRange = cacheFile->mLatRange;

      mWMSTexture = cs::graphics::TextureLoader::loadFromFile(cacheFile->mPath);
      mWMSTexture->Bind();
      mWMSTexture->SetWrapS(GL_CLAMP_TO_EDGE);
      mWMSTexture->SetWrapT(GL_CLAMP_TO_EDGE);
      mWMSTexture->Unbind();
      mWMSTextureUsed = true;
    } else {
      mWMSTextureUsed = false;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::requestUpdateBounds() {
  mUpdateLonLatRange = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<TimeInterval> TextureOverlayRenderer::getTimeIntervals() {
  return mTimeIntervals;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TextureOverlayRenderer::Do() {
  if (mUpdateLonLatRange) {
    updateLonLatRange();
    mUpdateLonLatRange = false;
  }

  if (mActiveWMSLayer && mActiveWMSLayer->getSettings().mTime.has_value()) {
    // Get the current time. Pre-fetch times are related to this.
    boost::posix_time::ptime time =
        cs::utils::convert::time::toPosix(mTimeControl->pSimulationTime.get());

    // Select WMS textures to be downloaded. If no pre-fetch is set, only sellect the texture for
    // the current timestep.
    for (int preFetch = -mPluginSettings->mPrefetchCount.get();
         preFetch <= mPluginSettings->mPrefetchCount.get(); preFetch++) {

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
        WebMapTextureLoader::Request request;
        request.mMaxSize  = mMaxSize;
        request.mStyle    = mStyle;
        request.mTime     = timeString;
        request.mLonRange = mLonRange;
        request.mLatRange = mLatRange;

        mTextureFilesBuffer.insert(
            std::pair<std::string, std::future<std::optional<WebMapTextureFile>>>(
                timeString, mTextureLoader.loadTextureAsync(*mActiveWMS, *mActiveWMSLayer, request,
                                mPluginSettings->mMapCache.get())));
      }
    }

    // Check whether the WMS textures are loaded to the disk.
    auto fileIt = mTextureFilesBuffer.begin();
    while (fileIt != mTextureFilesBuffer.end()) {
      if (fileIt->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        std::optional<WebMapTextureFile> file = fileIt->second.get();

        if (file.has_value()) {
          mLonRange = file->mLonRange;
          mLatRange = file->mLatRange;

          // Load WMS texture to memory
          mTexturesBuffer.insert(std::pair<std::string, std::future<std::optional<WebMapTexture>>>(
              fileIt->first, mTextureLoader.loadTextureFromFileAsync(file->mPath)));
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
        std::optional<WebMapTexture> texture = texIt->second.get();

        if (texture.has_value()) {
          mTextures.insert(std::pair<std::string, WebMapTexture>(texIt->first, texture.value()));
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
        mWMSTexture->UploadTexture(
            tex->second.mWidth, tex->second.mHeight, (void*)tex->second.mData, false);
        mCurrentTexture = timeString;
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
              tex->second.mWidth, tex->second.mHeight, (void*)tex->second.mData, false);
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
      mSolarSystem->pActiveBody.get()->getCenterName() != mCenterName) {
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
  // Only bind the enabled textures.
  if (mWMSTextureUsed) {
    mWMSTexture->Bind(GL_TEXTURE1);

    if (mSecondWMSTextureUsed) {
      m_pSurfaceShader->SetUniform(m_pSurfaceShader->GetUniformLocation("uFade"), mFade);
      mSecondWMSTexture->Bind(GL_TEXTURE2);
    }
  }

  m_pSurfaceShader->SetUniform(m_pSurfaceShader->GetUniformLocation("uDepthBuffer"), 0);
  m_pSurfaceShader->SetUniform(m_pSurfaceShader->GetUniformLocation("uFirstTexture"), 1);
  m_pSurfaceShader->SetUniform(m_pSurfaceShader->GetUniformLocation("uSecondTexture"), 2);

  m_pSurfaceShader->SetUniform(
      m_pSurfaceShader->GetUniformLocation("uUseFirstTexture"), mWMSTextureUsed);
  m_pSurfaceShader->SetUniform(
      m_pSurfaceShader->GetUniformLocation("uUseSecondTexture"), mSecondWMSTextureUsed);

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
  loc = m_pSurfaceShader->GetUniformLocation("uLatRange");
  glUniform2dv(loc, 1,
      glm::value_ptr(cs::utils::convert::toRadians(glm::dvec2(mLatRange[0], mLatRange[1]))));
  loc = m_pSurfaceShader->GetUniformLocation("uLonRange");
  glUniform2dv(loc, 1,
      glm::value_ptr(cs::utils::convert::toRadians(glm::dvec2(mLonRange[0], mLonRange[1]))));

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
  if (mWMSTextureUsed) {
    mWMSTexture->Unbind(GL_TEXTURE1);

    if (mSecondWMSTextureUsed) {
      mSecondWMSTexture->Unbind(GL_TEXTURE2);
    }
  }

  // Release shader
  m_pSurfaceShader->Release();

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glPopAttrib();
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TextureOverlayRenderer::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  oBoundingBox.SetBounds(mMinBounds.data(), mMaxBounds.data());
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
