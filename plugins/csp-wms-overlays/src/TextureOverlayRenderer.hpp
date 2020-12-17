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

/**
 * Class which gets a geo-referenced texture and overlays if onto the previous rendered scene.
 * Therefore it copies the depth buffer first. Second, in the shader it does an inverse projection
 * to get the cartesian coordinates. This coordinates are transformed to latitude and longitude to
 * do the lookup in the geo-referenced texture. The value is then overlayed on that pixel position.
 */
class TextureOverlayRenderer : public IVistaOpenGLDraw {
 public:
  /**
   * Constructor requires the SolarSystem to get the current active planet
   * to get the model matrix
   */
  TextureOverlayRenderer(std::shared_ptr<cs::core::SolarSystem> solarSystem,
      std::shared_ptr<cs::core::TimeControl>                    timeControl,
      std::shared_ptr<Plugin::Settings> const&                  pluginSettings);
  virtual ~TextureOverlayRenderer();

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::SimpleWMSBody const& settings);

  /// Set the active WMS data set.
  void setActiveWMS(std::shared_ptr<WebMapService> wms, std::shared_ptr<WebMapLayer> layer);

  /// Returns the time intervals of the active data set.
  std::vector<TimeInterval> getTimeIntervals();

  void SetBounds(std::array<double, 4> bounds);

  // ---------------------------------------
  // INTERFACE IMPLEMENTATION OF IVistaOpenGLDraw
  // ---------------------------------------
  virtual bool Do();
  virtual bool GetBoundingBox(VistaBoundingBox& bb);

 private:
  /// Delete stored textures.
  void clearTextures();

  std::shared_ptr<Plugin::Settings> mPluginSettings;
  Plugin::Settings::SimpleWMSBody   mSimpleWMSBodySettings;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  VistaGLSLShader* m_pSurfaceShader = nullptr; //! Vista GLSL shader object used for rendering

  std::vector<TimeInterval> mTimeIntervals; ///< Time intervals of data set.

  static const std::string SURFACE_GEOM; //! Code for the geometry shader
  static const std::string SURFACE_VERT; //! Code for the vertex shader
  static const std::string SURFACE_FRAG; //! Code for the fragment shader

  /**
   * Struct which stores the depth buffer and color buffer from the previous rendering (order)
   * on the GPU and pass it to the shaders for inverse transformations based on depth and screen
   * coordinates. Used to calculate texture coordinates for the overlay
   */
  struct GBufferData {
    VistaTexture* mDepthBuffer = nullptr;
    VistaTexture* mColorBuffer = nullptr;
  };

  std::unordered_map<VistaViewport*, GBufferData> mGBufferData; //! Store one buffer per viewport

  std::map<std::string, std::future<std::string>>    mTextureFilesBuffer;
  std::map<std::string, std::future<unsigned char*>> mTexturesBuffer;
  std::map<std::string, unsigned char*>              mTextures;
  std::vector<std::string>                           mWrongTextures;

  int                   mWidth  = 1024;
  int                   mHeight = 1024;
  std::array<double, 4> mLngLatBounds;

  std::shared_ptr<WebMapService> mActiveWMS;      ///< The active WMS.
  std::shared_ptr<WebMapLayer>   mActiveWMSLayer; ///< The active WMS layer.

  std::shared_ptr<VistaTexture> mWMSTexture;       ///< The WMS texture.
  std::shared_ptr<VistaTexture> mSecondWMSTexture; ///< Second WMS texture for time interpolation.
  bool                          mWMSTextureUsed;   ///< Whether to use the WMS texture.
  bool        mSecondWMSTextureUsed = false;       ///< Whether to use the second WMS texture.
  std::string mCurrentTexture;                     ///< Timestep of the current WMS texture.
  std::string mCurrentSecondTexture;               ///< Timestep of the second WMS texture.
  float       mFade;                               ///< Fading value between WMS textures.
  std::string mRequest;                            ///< WMS server request URL.
  std::string mFormat;                             ///< Time format style.
  Duration    mSampleDuration;                     ///< Sample rate of the current WMS data set.

  WebMapTextureLoader mTextureLoader;

  std::shared_ptr<cs::core::SolarSystem>
      mSolarSystem; //! Pointer to the CosmoScout solar system used to retrieve matrices
  std::shared_ptr<cs::core::TimeControl> mTimeControl;
};

} // namespace csp::wmsoverlays

#endif // TEXTURE_OVERLAY_RENDERER