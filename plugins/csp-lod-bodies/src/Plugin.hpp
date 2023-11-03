////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_PLUGIN_HPP
#define CSP_LOD_BODIES_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include "TileDataType.hpp"
#include "TileSourceWebMapService.hpp"

#include <glm/gtc/constants.hpp>
#include <vector>

class VistaOpenGLNode;

namespace csp::lodbodies {

class GLResources;
class LodBody;

/// This plugin provides planets with level of detail data. It uses separate image and elevation
/// data from either files or web map services to display the information onto the surface.
/// Multiple sources can be given at startup and they can be cycled through at runtime via the GUI.
/// The configuration is  done via the applications config file. See README.md for details.
class Plugin : public cs::core::PluginBase {
 public:
  /// The startup settings of the plugin.
  struct Settings {
    enum class ColorMappingType { eNone = 0, eHeight = 1, eSlope = 2 };
    enum class TerrainProjectionType { eHEALPix = 0, eLinear = 1, eHybrid = 2 };

    /// Select terrain projection interpolation type.
    cs::utils::DefaultProperty<TerrainProjectionType> mTerrainProjectionType{
        TerrainProjectionType::eHybrid};

    /// Specifies the amount of detail of the planet's surface. Should be in the range 1-100.
    cs::utils::DefaultProperty<float> mLODFactor{15.F};

    /// If set to true, the level-of-detail will be chosen automatically based on the current
    /// rendering performance.
    cs::utils::DefaultProperty<bool> mAutoLOD{true};

    /// If the level-of-detail is chosen automatically, it will be in this range.
    cs::utils::DefaultProperty<glm::vec2> mAutoLODRange{glm::vec2(10.F, 40.F)};

    /// The plugin will attempt to adjust the level-of-detail in such a way that the frame time
    /// stays between these values.
    cs::utils::DefaultProperty<glm::vec2> mAutoLODFrameTimeRange{glm::vec2(13.5F, 14.5F)};

    /// A multiplier for the brightness of the image channel.
    cs::utils::DefaultProperty<float> mTextureGamma{1.F};

    /// Enables or disables rendering of iso-altitude lines.
    cs::utils::DefaultProperty<bool> mEnableHeightlines{false};

    /// Enables or disables rendering of a latidude-longitude-grid.
    cs::utils::DefaultProperty<bool> mEnableLatLongGrid{false};

    /// If the latitude-longitude-grid is enabled, this function can be used to enable or disable
    /// rendering of grid labels.
    cs::utils::DefaultProperty<bool> mEnableLatLongGridLabels{false};

    /// Enable surface coloring based on slope or height.
    cs::utils::DefaultProperty<ColorMappingType> mColorMappingType{ColorMappingType::eNone};

    /// The file name for the colormap for height or slope coloring.
    cs::utils::DefaultProperty<std::string> mTerrainColorMap{""};

    /// The height range for the color mapping.
    cs::utils::DefaultProperty<glm::vec2> mHeightRange{glm::vec2(-8000.F, 12000.F)};

    /// The the slope range for the slope mapping.
    cs::utils::DefaultProperty<glm::vec2> mSlopeRange{glm::vec2(0.F, 0.25F * glm::pi<float>())};

    /// Enables or disables wireframe rendering of the planet.
    cs::utils::DefaultProperty<bool> mEnableWireframe{false};

    /// Enables or disables bounding box rendering of the planet's tiles.
    cs::utils::DefaultProperty<bool> mEnableBounds{false};

    /// Enables or disables debug coloring of the planet's tiles.
    cs::utils::DefaultProperty<bool> mEnableTilesDebug{false};

    /// If set to true, the level of detail and the frustum culling of the planet's tiles will not
    /// be updated anymore.
    cs::utils::DefaultProperty<bool> mEnableTilesFreeze{false};

    /// The maximum allowed colored tiles.
    cs::utils::DefaultProperty<uint32_t> mMaxGPUTilesColor{512};

    /// The maximum allowed elevation tiles.
    cs::utils::DefaultProperty<uint32_t> mMaxGPUTilesDEM{512};

    /// The vertex grid resolution used for terrain tiles.
    cs::utils::DefaultProperty<uint32_t> mTileResolutionDEM{128};

    /// The image channel resolution used for the tile textures.
    cs::utils::DefaultProperty<uint32_t> mTileResolutionIMG{512};

    /// Path to the map cache folder, can be absolute or relative to the cosmoscout executable.
    cs::utils::DefaultProperty<std::string> mMapCache{"map-cache"};

    /// A struct that represents a BRDF, given its source code and material properties.
    struct BRDF {
      std::string source; ///< The source code of the BRDF in GLSL-like form.
      std::unordered_map<std::string, float>
           properties; ///< The material properties as key-variables and values.
      bool operator==(BRDF const& other) const {
        return source == other.source && properties == other.properties;
      }
      bool operator!=(BRDF const& other) const {
        return !((*this) == other);
      }
    };

    /// A single data set containing either elevation or image data.
    struct Dataset {
      std::string mURL;        ///< The URL of the mapserver including the "SERVICE=wms" parameter.
      std::string mCopyright;  ///< The copyright holder of the data set (also shown in the UI).
      std::string mLayers;     ///< A comma,seperated list of WMS layers.
      uint32_t    mMaxLevel{}; ///< The maximum quadtree depth to load.
    };

    /// The startup settings for a planet.
    struct Body {
      std::string mActiveDemDataset; ///< The name of the currently active elevation data set.
      std::string mActiveImgDataset; ///< The name of the currently active image data set.
      std::map<std::string, Dataset>   mDemDatasets; ///< The data sets containing elevation data.
      std::map<std::string, Dataset>   mImgDatasets; ///< The data sets containing image data.
      cs::utils::DefaultProperty<BRDF> mBrdfHdr{
          ///< The BRDF used in HDR mode, with higher precedence than lighting mode.
          BRDF{"../share/resources/shaders/brdfs/lambert.glsl", {{"$rho", 1.0f}}}};
      cs::utils::DefaultProperty<BRDF>  mBrdfNonHdr{///< The BRDF used in lighting mode.
          BRDF{"../share/resources/shaders/brdfs/lambert_scaled.glsl", {{"$rho", 1.0f}}}};
      cs::utils::DefaultProperty<float> mAvgLinearImgIntensity{
          1.0f}; ///< The average intensity of the linear (!) image.
    };

    std::map<std::string, Body> mBodies; ///< A list of planets with their anchor names.
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();
  void onSave();
  void unregisterBody(std::string const& name);

  Settings::Body& getBodySettings(std::shared_ptr<LodBody> const& body) const;
  void setImageSource(std::shared_ptr<LodBody> const& body, std::string const& name) const;
  void setElevationSource(std::shared_ptr<LodBody> const& body, std::string const& name) const;

  std::shared_ptr<Settings>                       mPluginSettings = std::make_shared<Settings>();
  std::shared_ptr<GLResources>                    mGLResources;
  std::map<std::string, std::shared_ptr<LodBody>> mLodBodies;
  float                                           mNonAutoLod{};

  int mActiveObjectConnection = -1;
  int mOnLoadConnection       = -1;
  int mOnSaveConnection       = -1;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_PLUGIN_HPP
