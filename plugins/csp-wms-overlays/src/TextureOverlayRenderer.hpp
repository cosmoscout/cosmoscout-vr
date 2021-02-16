////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WMS_OVERLAYS_TEXTURE_OVERLAY_RENDERER_HPP
#define CSP_WMS_OVERLAYS_TEXTURE_OVERLAY_RENDERER_HPP

#include "Plugin.hpp"
#include "WebMapLayer.hpp"
#include "WebMapService.hpp"
#include "WebMapTextureLoader.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaGLSLShader.h>

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
  TextureOverlayRenderer(std::string center, std::shared_ptr<cs::core::SolarSystem> solarSystem,
      std::shared_ptr<cs::core::TimeControl> timeControl,
      std::shared_ptr<cs::core::Settings>    settings,
      std::shared_ptr<Plugin::Settings>      pluginSettings);
  virtual ~TextureOverlayRenderer();

  /// Returns the SPICE name of the body to which this renderer is assigned.
  std::string const& getCenter() const;

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::Body settings);

  /// Set the active WMS data set.
  void setActiveWMS(WebMapService const& wms, WebMapLayer const& layer);

  /// Clears the active WMS data set.
  void clearActiveWMS();

  /// Set the style that should be requested.
  void setStyle(std::string style);

  /// Requests to change the map bounds to values appropriate for the current observer perspective.
  /// The bounds will be updated the next time the Do() method is called.
  void requestUpdateBounds();

  /// The current map bounds of this overlay.
  /// This may be used for setting the bounds.
  cs::utils::Property<Bounds> pBounds;

  /// Interface implementation of IVistaOpenGLDraw
  virtual bool Do();
  virtual bool GetBoundingBox(VistaBoundingBox& bb);

 private:
  /// Delete stored textures.
  void clearTextures();

  /// Updates the longitude and latitude ranges according to the current viewport.
  void updateLonLatRange();

  /// Returns the manually set bounds if subsets are allowed by the active layer.
  /// Otherwise returns the default bounds of the layer.
  Bounds getBounds();

  /// Gets an appropriate Request object for the current state.
  WebMapTextureLoader::Request getRequest();

  /// Synchronously loads a texture for a time-independent map.
  void getTimeIndependentTexture(WebMapTextureLoader::Request const& request);

  std::shared_ptr<cs::core::Settings> mSettings;
  std::shared_ptr<Plugin::Settings>   mPluginSettings;
  Plugin::Settings::Body              mSimpleWMSOverlaySettings;
  std::string                         mCenterName;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  VistaGLSLShader mShader; ///< Vista GLSL shader object used for rendering

  static const std::string SURFACE_GEOM; ///< Code for the geometry shader
  static const std::string SURFACE_VERT; ///< Code for the vertex shader
  static const std::string SURFACE_FRAG; ///< Code for the fragment shader

  std::unordered_map<VistaViewport*, VistaTexture*>
      mDepthBufferData; ///< Store one buffer per viewport

  std::map<std::string, std::future<std::optional<WebMapTexture>>>
                                       mTexturesBuffer; ///< Stores all textures, for which the request ist still pending.
  std::map<std::string, WebMapTexture> mTextures; ///< Stores all successfully loaded textures.
  std::vector<std::string> mWrongTextures;        ///< Stores textures, for which loading failed.

  std::string mStyle; ///< Name of the currently active style.

  bool mUpdateLonLatRange = false; ///< Flag for updating the map bounds in the next update.

  std::optional<WebMapService> mActiveWMS;      ///< The active WMS.
  std::optional<WebMapLayer>   mActiveWMSLayer; ///< The active WMS layer.

  std::shared_ptr<VistaTexture> mWMSTexture;       ///< The WMS texture.
  std::shared_ptr<VistaTexture> mSecondWMSTexture; ///< Second WMS texture for time interpolation.
  bool                          mWMSTextureUsed;   ///< Whether to use the WMS texture.
  bool        mSecondWMSTextureUsed = false;       ///< Whether to use the second WMS texture.
  std::string mCurrentTexture;                     ///< Timestep of the current WMS texture.
  std::string mCurrentSecondTexture;               ///< Timestep of the second WMS texture.
  float       mFade;                               ///< Fading value between WMS textures.
  TimeInterval
      mCurrentInterval; ///< Used to save the current time format style and sample duration;

  WebMapTextureLoader mTextureLoader; ///< Loader used to request map textures.

  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::TimeControl> mTimeControl;

  std::array<float, 3> mMinBounds; ///< Lower Corner of the bounding volume for the planet.
  std::array<float, 3> mMaxBounds; ///< Upper Corner of the bounding volume for the planet.

  bool mShaderDirty        = true;
  int  mLightingConnection = -1;
  int  mHDRConnection      = -1;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_OVERLAYS_TEXTURE_OVERLAY_RENDERER_HPP
