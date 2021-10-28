////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WCS_OVERLAYS_TEXTURE_OVERLAY_RENDERER_HPP
#define CSP_WCS_OVERLAYS_TEXTURE_OVERLAY_RENDERER_HPP

#include "Plugin.hpp"
#include "WebCoverage.hpp"
#include "WebCoverageService.hpp"
#include "WebCoverageTextureLoader.hpp"

#include "../../../../src/cs-graphics/ColorMap.hpp"

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
class GuiManager;
} // namespace cs::core

namespace csp::wcsoverlays {

/// Class which gets a geo-referenced texture and overlays if onto the previous rendered scene.
/// Therefore it copies the depth buffer first. Second, in the shader it does an inverse projection
/// to get the cartesian coordinates. This coordinates are transformed to latitude and longitude to
/// do the lookup in the geo-referenced texture. The value is then overlayed on that pixel position.
class TextureOverlayRenderer : public IVistaOpenGLDraw {
 public:
  TextureOverlayRenderer(std::string center, std::shared_ptr<cs::core::SolarSystem> solarSystem,
      std::shared_ptr<cs::core::TimeControl> timeControl,
      std::shared_ptr<cs::core::Settings>    settings,
      std::shared_ptr<Plugin::Settings>      pluginSettings,
      std::shared_ptr<cs::core::GuiManager>  guiManager);
  ~TextureOverlayRenderer() override;

  /// Returns the SPICE name of the body to which this renderer is assigned.
  std::string const& getCenter() const;

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::Body settings);

  /// Set the active WCS data set.
  void setActiveWCS(WebCoverageService const& wcs, WebCoverage const& coverage);

  /// Clears the active WCS data set.
  void clearActiveWCS();

  /// Requests to change the map bounds to values appropriate for the current observer perspective.
  /// The bounds will be updated the next time the Do() method is called.
  void requestUpdateBounds();

  /// Set the transfer function used in the shader
  void setTransferFunction(const std::string& json);

  /// For multi-layer textures, set the texture layer to be rendered
  void setLayer(int layer);

  /// The current map bounds of this overlay.
  /// This may be used for setting the bounds.
  cs::utils::Property<Bounds> pBounds;

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
  Bounds getBounds() const;

  /// Gets an appropriate Request object for the current state.
  WebCoverageTextureLoader::Request getRequest();

  /// Synchronously loads a texture for a time-independent map.
  void getTimeIndependentTexture(WebCoverageTextureLoader::Request const& request);

  std::shared_ptr<cs::core::Settings> mSettings;
  std::shared_ptr<Plugin::Settings>   mPluginSettings;
  Plugin::Settings::Body              mSimpleWCSOverlaySettings;
  std::string                         mCenterName;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  VistaGLSLShader* m_pSurfaceShader = nullptr; //! Vista GLSL shader object used for rendering

  static const std::string SURFACE_GEOM; //! Code for the geometry shader
  static const std::string SURFACE_VERT; //! Code for the vertex shader
  static const std::string SURFACE_FRAG; //! Code for the fragment shader
  static const std::string COMPUTE;      //! Code for the compute shader

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

  GDALReader::GreyScaleTexture mTexture; //! The textured passed from outside via SetOverlayTexture
  int  mActiveLayer   = 1;     //! Active layer of the coverage (if the texture has more than one)
  bool mUpdateTexture = false; //! Flag if a texture upload is required

  std::unique_ptr<cs::graphics::ColorMap> mTransferFunction; //! Transfer function used in shader

  /// Stores all textures, for which the request ist still pending.
  std::map<std::string, std::future<std::optional<GDALReader::GreyScaleTexture>>> mTexturesBuffer;
  /// Stores all successfully loaded textures.
  std::map<std::string, GDALReader::GreyScaleTexture> mTextures;
  /// Stores textures, for which loading failed.
  std::vector<std::string> mWrongTextures;

  /// Flag for updating the coverage bounds in the next update.
  bool mUpdateLonLatRange = false;

  /// The active WCS.
  std::optional<WebCoverageService> mActiveWCS;
  /// The active WCS layer.
  std::optional<WebCoverage> mActiveWCSCoverage;

  /// Timestep of the current WCS texture.
  std::string mCurrentTexture;

  /// Used to save the current time format style and sample duration;
  TimeInterval mCurrentInterval;

  /// Loader used to request wcs textures.
  WebCoverageTextureLoader mTextureLoader;

  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::TimeControl> mTimeControl;
  std::shared_ptr<cs::core::GuiManager>  mGuiManager;

  /// Lower Corner of the bounding volume for the planet.
  std::array<float, 3> mMinBounds;
  /// Upper Corner of the bounding volume for the planet.
  std::array<float, 3> mMaxBounds;
};

} // namespace csp::wcsoverlays

#endif // CSP_WCS_OVERLAYS_TEXTURE_OVERLAY_RENDERER_HPP
