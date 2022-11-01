////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WMS_OVERLAYS_TEXTURE_OVERLAY_RENDERER_HPP
#define CSP_WMS_OVERLAYS_TEXTURE_OVERLAY_RENDERER_HPP

#include "Plugin.hpp"

#include "../../csl-ogc/src/wms/WebMapLayer.hpp"
#include "../../csl-ogc/src/wms/WebMapService.hpp"
#include "../../csl-ogc/src/wms/WebMapTextureLoader.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

#include <array>
#include <functional>
#include <future>
#include <unordered_map>

// FORWARD DEFINITIONS
class VistaGLSLShader;
class VistaOpenGLNode;
class VistaTexture;
class VistaViewport;

namespace cs::core {
class SolarSystem;
class TimeControl;
class Settings;
} // namespace cs::core

namespace csp::wmsoverlays {

/// Class which gets a geo-referenced texture and overlays if onto the previous rendered scene.
/// Therefore it copies the depth buffer first. Second, in the shader it does an inverse projection
/// to get the cartesian coordinates. This coordinates are transformed to latitude and longitude to
/// do the lookup in the geo-referenced texture. The value is then overlayed on that pixel position.
class TextureOverlayRenderer : public IVistaOpenGLDraw {
 public:
  TextureOverlayRenderer(std::string objectName, std::shared_ptr<cs::core::SolarSystem> solarSystem,
      std::shared_ptr<cs::core::TimeControl> timeControl,
      std::shared_ptr<cs::core::Settings>    settings,
      std::shared_ptr<Plugin::Settings>      pluginSettings);
  ~TextureOverlayRenderer() override;

  /// Returns the SPICE name of the body to which this renderer is assigned.
  std::string const& getObjectName() const;

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::Body settings);

  /// Set the active WMS data set.
  void setActiveWMS(csl::ogc::WebMapService const& wms, csl::ogc::WebMapLayer const& layer);

  /// Clears the active WMS data set.
  void clearActiveWMS();

  /// Set the style that should be requested.
  void setStyle(std::string style);

  /// Requests to change the map bounds to values appropriate for the current observer perspective.
  /// The bounds will be updated the next time the Do() method is called.
  void requestUpdateBounds();

  /// The current map bounds of this overlay.
  /// This may be used for setting the bounds.
  cs::utils::Property<csl::ogc::Bounds> pBounds;

  /// Interface implementation of IVistaOpenGLDraw
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& oBoundingBox) override;

 private:
  /// Delete stored textures.
  void clearTextures();

  /// Updates the longitude and latitude ranges according to the current viewport.
  void updateLonLatRange();

  /// Returns the manually set bounds if subsets are allowed by the active layer.
  /// Otherwise returns the default bounds of the layer.
  csl::ogc::Bounds getBounds();

  /// Gets an appropriate Request object for the current state.
  csl::ogc::WebMapTextureLoader::Request getRequest();

  /// Synchronously loads a texture for a time-independent map.
  void getTimeIndependentTexture(csl::ogc::WebMapTextureLoader::Request const& request);

  std::shared_ptr<cs::core::Settings> mSettings;
  std::shared_ptr<Plugin::Settings>   mPluginSettings;
  Plugin::Settings::Body              mSimpleWMSOverlaySettings;
  std::string                         mObjectName;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  /// Vista GLSL shader object used for rendering
  VistaGLSLShader mShader;

  /// Code for the geometry shader
  static const std::string SURFACE_GEOM;
  /// Code for the vertex shader
  static const std::string SURFACE_VERT;
  /// Code for the fragment shader
  static const std::string SURFACE_FRAG;

  /// Store one buffer per viewport
  std::unordered_map<VistaViewport*, VistaTexture> mDepthBufferData;

  /// Stores all textures, for which the request ist still pending.
  std::map<std::string, std::future<std::optional<csl::ogc::WebMapTexture>>> mTexturesBuffer;
  /// Stores all successfully loaded textures.
  std::map<std::string, csl::ogc::WebMapTexture> mTextures;
  /// Stores textures, for which loading failed.
  std::vector<std::string> mWrongTextures;

  /// Name of the currently active style.
  std::string mStyle;

  /// Flag for updating the map bounds in the next update.
  bool mUpdateLonLatRange = false;

  /// The active WMS.
  std::optional<csl::ogc::WebMapService> mActiveWMS;
  /// The active WMS layer.
  std::optional<csl::ogc::WebMapLayer> mActiveWMSLayer;

  /// The WMS texture.
  VistaTexture mWMSTexture;
  /// Second WMS texture for time interpolation.
  VistaTexture mSecondWMSTexture;
  /// Whether to use the WMS texture.
  bool mWMSTextureUsed{};
  /// Whether to use the second WMS texture.
  bool mSecondWMSTextureUsed = false;
  /// Timestep of the current WMS texture.
  std::string mCurrentTexture;
  /// Timestep of the second WMS texture.
  std::string mCurrentSecondTexture;
  /// Fading value between WMS textures.
  float mFade{};
  /// Used to save the current time format style and sample duration;
  csl::ogc::TimeInterval mCurrentInterval;

  /// Loader used to request map textures.
  csl::ogc::WebMapTextureLoader mTextureLoader;

  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::TimeControl> mTimeControl;

  /// Lower Corner of the bounding volume for the planet.
  glm::vec3 mMinBounds;
  /// Upper Corner of the bounding volume for the planet.
  glm::vec3 mMaxBounds;

  bool mShaderDirty        = true;
  int  mLightingConnection = -1;
  int  mHDRConnection      = -1;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_OVERLAYS_TEXTURE_OVERLAY_RENDERER_HPP
