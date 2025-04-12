////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// Plugin Includes
#include "TextureOverlayRenderer.hpp"

#include "logger.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-scene/IntersectableObject.hpp"
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
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

// Standard includes
#include <functional>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>

namespace csp::wmsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

TextureOverlayRenderer::TextureOverlayRenderer(std::string objectName,
    std::shared_ptr<cs::core::SolarSystem>                 solarSystem,
    std::shared_ptr<cs::core::TimeControl>                 timeControl,
    std::shared_ptr<cs::core::Settings> settings, std::shared_ptr<Plugin::Settings> pluginSettings)
    : mSettings(std::move(settings))
    , mPluginSettings(std::move(pluginSettings))
    , mObjectName(std::move(objectName))
    , mWMSTexture(GL_TEXTURE_2D)
    , mSecondWMSTexture(GL_TEXTURE_2D)
    , mSolarSystem(std::move(solarSystem))
    , mTimeControl(std::move(timeControl)) {

  auto object = mSolarSystem->getObject(mObjectName);
  mMinBounds  = -object->getRadii();
  mMaxBounds  = object->getRadii();

  // create textures ---------------------------------------------------------
  for (auto const& viewport : GetVistaSystem()->GetDisplayManager()->GetViewports()) {
    // Texture for previous renderer depth buffer
    const auto [buffer, success] =
        mDepthBufferData.try_emplace(viewport.second, GL_TEXTURE_RECTANGLE);
    if (success) {
      buffer->second.Bind();
      buffer->second.SetWrapS(GL_CLAMP);
      buffer->second.SetWrapT(GL_CLAMP);
      buffer->second.SetMinFilter(GL_NEAREST);
      buffer->second.SetMagFilter(GL_NEAREST);
      buffer->second.Unbind();
    }
  }

  mWMSTexture.Bind();
  mWMSTexture.SetWrapS(GL_CLAMP_TO_EDGE);
  mWMSTexture.SetWrapT(GL_CLAMP_TO_EDGE);
  mWMSTexture.Unbind();
  mSecondWMSTexture.Bind();
  mSecondWMSTexture.SetWrapS(GL_CLAMP_TO_EDGE);
  mSecondWMSTexture.SetWrapT(GL_CLAMP_TO_EDGE);
  mSecondWMSTexture.Unbind();

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::ePlanets) + 10);

  pBounds.connect([this](Bounds const& value) {
    clearTextures();
    if (mActiveWMSLayer->getSettings().mTimeIntervals.empty()) {
      WebMapTextureLoader::Request request = getRequest();
      request.mBounds                      = value;
      getTimeIndependentTexture(request);
    }
  });

  mPluginSettings->mMaxTextureSize.connect([this](int value) {
    clearTextures();
    if (mActiveWMSLayer->getSettings().mTimeIntervals.empty()) {
      WebMapTextureLoader::Request request = getRequest();
      request.mMaxSize                     = value;
      getTimeIndependentTexture(request);
    }
  });

  // Recreate the shader if lighting or HDR rendering mode are toggled.
  mLightingConnection = mSettings->mGraphics.pEnableLighting.connect(
      [this](bool /*unused*/) { mShaderDirty = true; });
  mHDRConnection =
      mSettings->mGraphics.pEnableHDR.connect([this](bool /*unused*/) { mShaderDirty = true; });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TextureOverlayRenderer::~TextureOverlayRenderer() {
  mSettings->mGraphics.pEnableLighting.disconnect(mLightingConnection);
  mSettings->mGraphics.pEnableHDR.disconnect(mHDRConnection);

  clearTextures();

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& TextureOverlayRenderer::getObjectName() const {
  return mObjectName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::configure(Plugin::Settings::Body settings) {
  mSimpleWMSOverlaySettings = std::move(settings);
  pBounds                   = mSimpleWMSOverlaySettings.mActiveBounds.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::setActiveWMS(WebMapService const& wms, WebMapLayer const& layer) {
  clearActiveWMS();

  mActiveWMS.emplace(wms);
  mActiveWMSLayer.emplace(layer);

  if (mActiveWMSLayer && mActiveWMSLayer->isRequestable()) {
    if (!mActiveWMSLayer->getSettings().mTimeIntervals.empty()) {
      mCurrentInterval = mActiveWMSLayer->getSettings().mTimeIntervals.at(0);
    } else {
      getTimeIndependentTexture(getRequest());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::clearActiveWMS() {
  clearTextures();

  mWMSTextureUsed       = false;
  mSecondWMSTextureUsed = false;
  mStyle                = "";

  mActiveWMS.reset();
  mActiveWMSLayer.reset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::setStyle(std::string style) {
  if (mStyle != style) {
    mStyle = std::move(style);

    clearTextures();
    if (mActiveWMSLayer->getSettings().mTimeIntervals.empty()) {
      getTimeIndependentTexture(getRequest());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::clearTextures() {
  mTextures.clear();
  mTexturesBuffer.clear();
  mWrongTextures.clear();

  mCurrentTexture       = "";
  mCurrentSecondTexture = "";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::updateLonLatRange() {
  if (!mActiveWMS || !mActiveWMSLayer) {
    return;
  }

  VistaProjection::VistaProjectionProperties* projectionProperties =
      GetVistaSystem()
          ->GetDisplayManager()
          ->GetCurrentRenderInfo()
          ->m_pViewport->GetProjection()
          ->GetProjectionProperties();

  float posX, posY, posZ;
  projectionProperties->GetProjPlaneMidpoint(posX, posY, posZ);
  double left, right, bottom, top;
  projectionProperties->GetProjPlaneExtents(left, right, bottom, top);

  // Get the intersections of the camera rays at the corners of the screen with the body.
  auto object        = mSolarSystem->getObject(mObjectName);
  auto intersectable = object->getIntersectableObject();

  if (!intersectable) {
    return;
  }

  std::array<std::pair<bool, glm::dvec3>, 4> intersections;
  intersections[0].first = intersectable->getIntersection(
      glm::dvec3(0, 0, 0), glm::normalize(glm::dvec3(left, top, posZ)), intersections[0].second);
  intersections[1].first = intersectable->getIntersection(
      glm::dvec3(0, 0, 0), glm::normalize(glm::dvec3(left, bottom, posZ)), intersections[1].second);
  intersections[2].first = intersectable->getIntersection(glm::dvec3(0, 0, 0),
      glm::normalize(glm::dvec3(right, bottom, posZ)), intersections[2].second);
  intersections[3].first = intersectable->getIntersection(
      glm::dvec3(0, 0, 0), glm::normalize(glm::dvec3(right, top, posZ)), intersections[3].second);

  if (!std::all_of(intersections.begin(), intersections.end(),
          [](auto intersection) { return intersection.first; })) {
    // The body is not visible in all four corners of the screen.
    // For now this results in using the maximum bounds of the map.
    pBounds = mActiveWMSLayer->getSettings().mBounds;
  } else {
    // All four corners of the screen show the body.
    // The intersection points can be converted to longitude and latitude.
    Bounds currentBounds;

    glm::dvec3                radii = object->getRadii();
    std::array<glm::dvec2, 4> screenBounds{};
    for (int i = 0; i < 4; i++) {
      screenBounds[i] = cs::utils::convert::cartesianToLngLat(intersections[i].second, radii);
      screenBounds[i] = cs::utils::convert::toDegrees(screenBounds[i]);
    }

    currentBounds.mMinLon = screenBounds[0][0];
    currentBounds.mMaxLon = screenBounds[0][0];

    // Determine the minimum and maximum longitude.
    // To do so, the edges between neighboring corners are examined and classified as one of four
    // categories. Depending on the category the longitude range can be updated.
    // Also save the lengths of the edges for later (lonDiffs).
    // Uses counterclockwise winding order.
    std::array<double, 4> lonDiffs{};
    double                offset = 0;
    for (int i = 1; i < 5; i++) {
      if (screenBounds[i % 4][0] > screenBounds[i - 1][0]) {
        if (screenBounds[i % 4][0] - screenBounds[i - 1][0] < 180) {
          // 0  90  180 270 360
          // | x---x |   |   |
          //   1   2
          // West to east, dateline is not crossed
          currentBounds.mMaxLon = std::max(currentBounds.mMaxLon, screenBounds[i % 4][0] + offset);
          lonDiffs[i - 1]       = screenBounds[i % 4][0] - screenBounds[i - 1][0];
        } else {
          // 0  90  180 270 360
          // --x |   |   | x--
          //   1           2
          // East to west, dateline is crossed
          currentBounds.mMinLon = std::min(currentBounds.mMinLon + 360, screenBounds[i % 4][0]);
          currentBounds.mMaxLon = currentBounds.mMaxLon + 360;
          lonDiffs[i - 1]       = screenBounds[i % 4][0] - (screenBounds[i - 1][0] + 360);
        }
      } else {
        if (screenBounds[i - 1][0] - screenBounds[i % 4][0] < 180) {
          // 0  90  180 270 360
          // | x---x |   |   |
          //   2   1
          // East to west, dateline is not crossed
          currentBounds.mMinLon = std::min(currentBounds.mMinLon, screenBounds[i % 4][0] + offset);
          lonDiffs[i - 1]       = screenBounds[i % 4][0] - screenBounds[i - 1][0];
        } else {
          // 0  90  180 270 360
          // --x |   |   | x--
          //   2           1
          // West to East, dateline is crossed
          currentBounds.mMaxLon = std::max(currentBounds.mMaxLon, screenBounds[i % 4][0] + 360);
          offset                = 360;
          lonDiffs[i - 1]       = (screenBounds[i % 4][0] + 360) - screenBounds[i - 1][0];
        }
      }
    }
    if (currentBounds.mMaxLon > 360) {
      currentBounds.mMinLon -= 360;
      currentBounds.mMaxLon -= 360;
    }

    std::array<double, 4> lats{};
    std::transform(screenBounds.begin(), screenBounds.end(), lats.begin(),
        [](glm::dvec2 corner) { return corner[1]; });

    currentBounds.mMinLat = *std::min_element(lats.begin(), lats.end());
    currentBounds.mMaxLat = *std::max_element(lats.begin(), lats.end());

    // Check if the longitude range spans the whole earth, which would mean that one of the poles is
    // visible. >= 270 is used instead of >= 360 to prevent floating point errors.
    // As long as no pole is visible the maximum range should be 180 degrees, so this check can not
    // result in false positives.
    if (currentBounds.mMaxLon - currentBounds.mMinLon >= 270) {
      // 360 degree ranges other than [-180, 180] result in problems on some servers.
      currentBounds.mMinLon = -180.;
      currentBounds.mMaxLon = 180.;
      if (std::all_of(lonDiffs.begin(), lonDiffs.end(), [](double diff) { return diff > 0; })) {
        // West to east => north pole is visible
        currentBounds.mMaxLat = 90;
      } else if (std::all_of(
                     lonDiffs.begin(), lonDiffs.end(), [](double diff) { return diff < 0; })) {
        // East to west => south pole is visible
        currentBounds.mMinLat = -90;
      } else {
        logger().warn("Could not determine which pole is visible");
      }
    }

    pBounds = currentBounds;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Bounds TextureOverlayRenderer::getBounds() {
  if (mActiveWMSLayer && mActiveWMSLayer->getSettings().mNoSubsets) {
    return mActiveWMSLayer->getSettings().mBounds;
  }
  return pBounds.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapTextureLoader::Request TextureOverlayRenderer::getRequest() {
  WebMapTextureLoader::Request request;
  request.mMaxSize = mPluginSettings->mMaxTextureSize.get();
  request.mStyle   = mStyle;
  request.mBounds  = getBounds();
  return request;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TextureOverlayRenderer::getTimeIndependentTexture(
    WebMapTextureLoader::Request const& request) {
  if (mActiveWMSLayer && mActiveWMSLayer->isRequestable()) {
    std::optional<WebMapTexture> texture = mTextureLoader.loadTexture(*mActiveWMS, *mActiveWMSLayer,
        request, mPluginSettings->mMapCache.get(),
        request.mBounds == mActiveWMSLayer->getSettings().mBounds);
    if (texture.has_value()) {
      mWMSTexture.UploadTexture(texture->mWidth, texture->mHeight, texture->mData.get(), false);
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

bool TextureOverlayRenderer::Do() {
  if (mUpdateLonLatRange) {
    updateLonLatRange();
    mUpdateLonLatRange = false;
  }

  if (mShaderDirty) {
    mShader = VistaGLSLShader();

    std::string defines = "#version 440\n";

    if (mSettings->mGraphics.pEnableHDR.get()) {
      defines += "#define ENABLE_HDR\n";
    }

    if (mSettings->mGraphics.pEnableLighting.get()) {
      defines += "#define ENABLE_LIGHTING\n";
    }

    mShader.InitGeometryShaderFromString(SURFACE_GEOM);
    mShader.InitVertexShaderFromString(SURFACE_VERT);
    mShader.InitFragmentShaderFromString(defines + SURFACE_FRAG);
    mShader.Link();

    mShaderDirty = false;
  }

  if (mSolarSystem->pActiveObject.get() == nullptr ||
      mSolarSystem->pActiveObject != mSolarSystem->getObject(mObjectName) ||
      !mActiveWMS.has_value() || !mActiveWMSLayer.has_value()) {
    return false;
  }

  if (mActiveWMSLayer && !mActiveWMSLayer->getSettings().mTimeIntervals.empty()) {
    // Get the current time. Pre-fetch times are related to this.
    std::chrono::utc_clock::time_point time =
        cs::utils::convert::time::toUTC(mTimeControl->pSimulationTime.get());

    // Select WMS textures to be downloaded. If no pre-fetch is set, only sellect the texture for
    // the current timestep.
    for (int preFetch = -mPluginSettings->mPrefetchCount.get();
        preFetch <= mPluginSettings->mPrefetchCount.get(); preFetch++) {

      // Get the start time of the WMS sample.
      std::chrono::utc_clock::time_point sampleStartTime =
          time + (mCurrentInterval.mSampleDuration * preFetch);

      sampleStartTime = std::chrono::floor<std::chrono::milliseconds>(sampleStartTime);
      bool inInterval = utils::timeInIntervals(
          sampleStartTime, mActiveWMSLayer->getSettings().mTimeIntervals, mCurrentInterval);

      // Create identifier for the sample start time.
      std::string timeString = utils::timeToString(mCurrentInterval.mFormat, sampleStartTime);

      auto requestedTexture = mTexturesBuffer.find(timeString);
      auto loadedTexture    = mTextures.find(timeString);
      auto wrongTexture     = std::ranges::find(mWrongTextures, timeString);

      // Only load textures that aren't stored yet.
      if (requestedTexture == mTexturesBuffer.end() && loadedTexture == mTextures.end() &&
          wrongTexture == mWrongTextures.end() && inInterval) {
        // Load WMS texture.
        WebMapTextureLoader::Request request;
        request.mMaxSize = mPluginSettings->mMaxTextureSize.get();
        request.mStyle   = mStyle;
        request.mTime    = timeString;
        request.mBounds  = getBounds();

        mTexturesBuffer.emplace(
            timeString, mTextureLoader.loadTextureAsync(*mActiveWMS, *mActiveWMSLayer, request,
                            mPluginSettings->mMapCache.get(),
                            request.mBounds == mActiveWMSLayer->getSettings().mBounds));
      }
    }

    // Check whether the WMS textures are loaded to the memory.
    auto texIt = mTexturesBuffer.begin();
    while (texIt != mTexturesBuffer.end()) {
      if (texIt->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        std::optional<WebMapTexture> texture = texIt->second.get();

        if (texture.has_value()) {
          mTextures.emplace(texIt->first, std::move(texture.value()));
        } else {
          mWrongTextures.emplace_back(texIt->first);
        }

        texIt = mTexturesBuffer.erase(texIt);
      } else {
        ++texIt;
      }
    }

    // Get the current time.
    time = cs::utils::convert::time::toUTC(mTimeControl->pSimulationTime.get());

    std::chrono::utc_clock::time_point sampleStartTime{
        std::chrono::floor<std::chrono::milliseconds>(time)};

    bool inInterval = utils::timeInIntervals(
        sampleStartTime, mActiveWMSLayer->getSettings().mTimeIntervals, mCurrentInterval);

    // Create identifier for the sample start time.
    std::string timeString = utils::timeToString(mCurrentInterval.mFormat, sampleStartTime);

    // Find the current texture.
    auto tex = mTextures.find(timeString);

    // Use Wms texture inside the interval.
    if (inInterval && tex != mTextures.end()) {
      // Only update if we have a new texture.
      if (mCurrentTexture != timeString) {
        mWMSTextureUsed = true;
        mWMSTexture.UploadTexture(
            tex->second.mWidth, tex->second.mHeight, tex->second.mData.get(), false);
        mCurrentTexture = timeString;
      }
    } // Use default planet texture instead.
    else {
      mWMSTextureUsed = false;
      mCurrentTexture = "";
    }

    if (!mWMSTextureUsed || !mPluginSettings->mEnableInterpolation.get() ||
        mCurrentInterval.mSampleDuration.count() == 0) {
      mSecondWMSTextureUsed = false;
      mCurrentSecondTexture = "";
    } // Create fading between Wms textures when interpolation is enabled.
    else {
      std::chrono::utc_clock::time_point sampleAfter =
          sampleStartTime + mCurrentInterval.mSampleDuration;
      bool isAfterInInterval = utils::timeInIntervals(
          sampleAfter, mActiveWMSLayer->getSettings().mTimeIntervals, mCurrentInterval);

      // Find texture for the following sample.
      tex = mTextures.find(utils::timeToString(mCurrentInterval.mFormat, sampleAfter));

      if (isAfterInInterval && tex != mTextures.end()) {
        // Only update if we ha a new second texture.
        if (mCurrentSecondTexture != utils::timeToString(mCurrentInterval.mFormat, sampleAfter)) {
          mSecondWMSTexture.UploadTexture(
              tex->second.mWidth, tex->second.mHeight, tex->second.mData.get(), false);
          mCurrentSecondTexture = utils::timeToString(mCurrentInterval.mFormat, sampleAfter);
          mSecondWMSTextureUsed = true;
        }
        // Interpolate fade value between the 2 WMS textures.
        mFade = static_cast<float>(
            std::chrono::duration<double>(sampleAfter - time).count() /
            std::chrono::duration<double>(sampleAfter - sampleStartTime).count());
      }
    }
  }

  // save current lighting and material state of the OpenGL state machine
  glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT);
  glEnable(GL_TEXTURE_2D);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  glEnable(GL_BLEND);

  // copy depth buffer from previous rendering
  std::array<GLint, 4> iViewport{};
  glGetIntegerv(GL_VIEWPORT, iViewport.data());

  auto* viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  VistaTexture& depthBuffer = mDepthBufferData.at(viewport);

  depthBuffer.Bind();
  glCopyTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT, iViewport[0], iViewport[1],
      iViewport[2], iViewport[3], 0);

  auto object    = mSolarSystem->getObject(mObjectName);
  auto radii     = object->getRadii();
  auto transform = object->getObserverRelativeTransform();

  // get matrices and related values
  std::array<GLfloat, 16> glMatV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matV = glm::make_mat4x4(glMatV.data());
  auto matP = glm::make_mat4x4(glMatP.data());

  glm::dmat4 matInvMVP = glm::inverse(glm::dmat4(matP) * glm::dmat4(matV) * transform);

  // Bind shader before draw
  mShader.Bind();

  // Only bind the enabled textures.
  depthBuffer.Bind(GL_TEXTURE0);
  if (mWMSTextureUsed) {
    mWMSTexture.Bind(GL_TEXTURE1);

    if (mSecondWMSTextureUsed) {
      mShader.SetUniform(mShader.GetUniformLocation("uFade"), mFade);
      mSecondWMSTexture.Bind(GL_TEXTURE2);
    }
  }

  mShader.SetUniform(mShader.GetUniformLocation("uDepthBuffer"), 0);
  mShader.SetUniform(mShader.GetUniformLocation("uFirstTexture"), 1);
  mShader.SetUniform(mShader.GetUniformLocation("uSecondTexture"), 2);

  mShader.SetUniform(mShader.GetUniformLocation("uUseFirstTexture"), mWMSTextureUsed);
  mShader.SetUniform(mShader.GetUniformLocation("uUseSecondTexture"), mSecondWMSTextureUsed);

  GLint loc = mShader.GetUniformLocation("uMatInvMVP");
  glUniformMatrix4dv(loc, 1, GL_FALSE, glm::value_ptr(matInvMVP));

  // Double precision bounds
  loc = mShader.GetUniformLocation("uLatRange");
  glUniform2dv(loc, 1,
      glm::value_ptr(
          cs::utils::convert::toRadians(glm::dvec2(getBounds().mMinLat, getBounds().mMaxLat))));
  loc = mShader.GetUniformLocation("uLonRange");
  glUniform2dv(loc, 1,
      glm::value_ptr(
          cs::utils::convert::toRadians(glm::dvec2(getBounds().mMinLon, getBounds().mMaxLon))));

  glm::vec3 sunDirection(1, 0, 0);
  float     sunIlluminance(1.F);
  float     ambientBrightness(mSettings->mGraphics.pAmbientBrightness.get());

  if (object == mSolarSystem->getSun()) {
    // If the overlay is on the sun, we have to calculate the lighting differently.
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

    sunDirection = glm::normalize(
        glm::inverse(transform) * glm::dvec4(mSolarSystem->getSunDirection(transform[3]), 0.0));
  }

  mShader.SetUniform(mShader.GetUniformLocation("uSunDirection"), sunDirection[0], sunDirection[1],
      sunDirection[2]);
  mShader.SetUniform(mShader.GetUniformLocation("uSunIlluminance"), sunIlluminance);
  mShader.SetUniform(mShader.GetUniformLocation("uAmbientBrightness"), ambientBrightness);

  // provide radii to shader
  mShader.SetUniform(mShader.GetUniformLocation("uRadii"), static_cast<float>(radii[0]),
      static_cast<float>(radii[1]), static_cast<float>(radii[2]));

  int depthBits = 0;
  glGetIntegerv(GL_DEPTH_BITS, &depthBits);

  // Dummy draw
  glDrawArrays(GL_POINTS, 0, 1);

  depthBuffer.Unbind(GL_TEXTURE0);
  if (mWMSTextureUsed) {
    mWMSTexture.Unbind(GL_TEXTURE1);

    if (mSecondWMSTextureUsed) {
      mSecondWMSTexture.Unbind(GL_TEXTURE2);
    }
  }

  // Release shader
  mShader.Release();

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glPopAttrib();
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TextureOverlayRenderer::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  oBoundingBox.SetBounds(glm::value_ptr(mMinBounds), glm::value_ptr(mMaxBounds));
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
