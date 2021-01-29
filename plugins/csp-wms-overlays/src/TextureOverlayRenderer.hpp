#ifndef TEXTURE_OVERLAY_RENDERER
#define TEXTURE_OVERLAY_RENDERER

#include "Plugin.hpp"
#include "WebMapLayer.hpp"
#include "WebMapService.hpp"
#include "WebMapTextureLoader.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaMath/VistaBoundingBox.h>

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
      std::shared_ptr<cs::core::TimeControl>   timeControl,
      std::shared_ptr<Plugin::Settings> const& pluginSettings);
  virtual ~TextureOverlayRenderer();

  /// Returns the SPICE name of the body to which this renderer is assigned.
  std::string getCenter() const;

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::Body const& settings);

  /// Set the active WMS data set.
  void setActiveWMS(WebMapService const& wms, WebMapLayer const& layer);

  /// Clears the active WMS data set.
  void clearActiveWMS();

  /// Set the style that should be requested.
  void setStyle(std::string style);

  /// Requests to change the map bounds to values appropriate for the current observer perspective.
  /// The bounds will be updated the next time the Do() method is called.
  void requestUpdateBounds();

  /// Set the current map bounds.
  void setBounds(Bounds const& bounds);

  /// The current map bounds of this overlay.
  /// Consider this to be read-only. Use setBounds() for setting instead.
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

  /// Synchronously loads a texture for a time-independent map.
  void getTimeIndependentTexture();

  std::shared_ptr<Plugin::Settings> mPluginSettings;
  Plugin::Settings::Body            mSimpleWMSOverlaySettings;
  std::string                       mCenterName;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  VistaGLSLShader* m_pSurfaceShader = nullptr; ///< Vista GLSL shader object used for rendering

  static const std::string SURFACE_GEOM; ///< Code for the geometry shader
  static const std::string SURFACE_VERT; ///< Code for the vertex shader
  static const std::string SURFACE_FRAG; ///< Code for the fragment shader

  /// Struct which stores the depth buffer and color buffer from the previous rendering (order)
  /// on the GPU and pass it to the shaders for inverse transformations based on depth and screen
  /// coordinates. Used to calculate texture coordinates for the overlay
  struct GBufferData {
    VistaTexture* mDepthBuffer = nullptr;
    VistaTexture* mColorBuffer = nullptr;
  };

  std::unordered_map<VistaViewport*, GBufferData> mGBufferData; ///< Store one buffer per viewport

  std::map<std::string, std::future<std::optional<WebMapTexture>>>
      mTexturesBuffer; ///< Stores all textures, for which the request ist still pending.
  std::map<std::string, WebMapTexture> mTextures; ///< Stores all successfully loaded textures.
  std::vector<std::string> mWrongTextures;        ///< Stores textures, for which loading failed.

  std::string mStyle;   ///< Name of the currently active style.

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
};

} // namespace csp::wmsoverlays

#endif // TEXTURE_OVERLAY_RENDERER
