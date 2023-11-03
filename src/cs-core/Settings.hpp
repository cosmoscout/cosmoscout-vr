////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_SETTINGS_HPP
#define CS_CORE_SETTINGS_HPP

#include "cs_core_export.hpp"

#include "../cs-graphics/HDRBuffer.hpp"
#include "../cs-graphics/ToneMappingNode.hpp"
#include "../cs-scene/CelestialObject.hpp"
#include "../cs-utils/DefaultProperty.hpp"
#include "../cs-utils/ObservableMap.hpp"
#include "../cs-utils/utils.hpp"
#include "EclipseShadowReceiver.hpp"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <cstdint>
#include <deque>
#include <exception>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace nlohmann {

// A partial template specialization for serialization and deserialization of glm::*vec*. This
// allows using glm's vector types as settings elements.
template <int C, typename T, glm::qualifier Q>
struct adl_serializer<glm::vec<C, T, Q>> {
  static void to_json(json& j, glm::vec<C, T, Q> const& opt) {
    j = json::array();
    for (int i = 0; i < C; ++i) {
      j[i] = opt[i];
    }
  }

  static void from_json(json const& j, glm::vec<C, T, Q>& opt) {
    for (int i = 0; i < C; ++i) {
      j.at(i).get_to(opt[i]);
    }
  }
};

// A partial template specialization for serialization and deserialization of glm::*qua*. This
// allows using glm's quaternion types as settings elements.
// A very weird thing about glm is that it mixes the order of quaternion members. The constructor
// expects qua(w, x, y, z), the accessor returns qua[0] = x, qua[1] = y, qua[2] = z, qua[3] = w.
// In the settings of CosmoScout VR and in the user interface we always attempt to show w as the
// last component.
template <typename T, glm::qualifier Q>
struct adl_serializer<glm::qua<T, Q>> {
  static void to_json(json& j, glm::qua<T, Q> const& opt) {
    j = {opt[0], opt[1], opt[2], opt[3]};
  }

  static void from_json(json const& j, glm::qua<T, Q>& opt) {
    j.at(0).get_to(opt[0]);
    j.at(1).get_to(opt[1]);
    j.at(2).get_to(opt[2]);
    j.at(3).get_to(opt[3]);
  }
};

// A partial template specialization for cs::utils::Property. This allows using our Properties as
// settings elements.
template <typename T>
struct adl_serializer<cs::utils::Property<T>> {
  static void to_json(json& j, cs::utils::Property<T> const& opt) {
    j = opt.get();
  }

  static void from_json(json const& j, cs::utils::Property<T>& opt) {
    opt = j.get<T>();
  }
};

} // namespace nlohmann

namespace cs::utils {
CS_CORE_EXPORT void from_json(nlohmann::json const&                                j,
    ObservableMap<std::string, std::shared_ptr<const cs::scene::CelestialObject>>& o);
CS_CORE_EXPORT void to_json(nlohmann::json&                                              j,
    ObservableMap<std::string, std::shared_ptr<const cs::scene::CelestialObject>> const& o);
} // namespace cs::utils

namespace cs::core {

/// Most of CosmoScout VR's configuration is done with one huge JSON file. This contains some global
/// options and settings for each plugin. The available global options are defined below. The
/// per-plugin settings are defined in each and every plugin (but they are also stored in this
/// class in the mPlugins member).
/// When the settings are loaded from file (using the read() method), all values in the class are
/// updated accordingly.
/// There are basically four setting types:
/// std::optional<T>:   Settings which have a std::optional type can be defined in the JSON file but
///                     do not have to. If they are not present in the JSON file, they will be set
///                     to std::nullopt. When settings are reloaded, you have to check whether the
///                     value has changed. When saving, it will only be written if it is not
///                     currently set to std::nullopt.
/// Property<T>:        This is a mandatory element; omitting it in the JSON file will lead to an
///                     error. You can connect to the onChange() signal in order to be notified when
///                     the value changes. This, for example, could be caused by modifing a
///                     corresponding widget in the user interface or by reloading the settings.
/// DefaultProperty<T>: Similar to the Property<T> but not mandatory. When reading from file, it
///                     will be set to its default state if it's not present in the file. On save,
///                     it will not be written to file when it's currently in its default state.
/// Everything else:    Everything else is considered to be mandatory.
class CS_CORE_EXPORT Settings {
 public:
  // -----------------------------------------------------------------------------------------------

  /// This Signal is emitted when the settings are reloaded from file. You can connect a function
  /// to check whether something has changed compared to the last settings state. The very first
  /// onLoad will be emitted before any Plugin or core class is initialized, so don't rely on that
  /// one. Please catch all exceptions inside your handler, else other handlers will not be called
  /// properly!
  utils::Signal<> const& onLoad() const;

  /// This signal is emitted before the settings are written to file. You can use this to update any
  /// fields according to the current scene state. Please catch all exceptions inside your handler,
  /// else other handlers will not be called properly!
  utils::Signal<> const& onSave() const;

  /// Initializes all members from a given JSON file. Once reading finished, the onLoad signal will
  /// be emitted.
  void loadFromFile(std::string const& fileName);

  /// Initializes all members from a given JSON object. Once reading finished, the onLoad signal
  /// will be emitted.
  void loadFromJson(std::string const& json);

  /// Writes the current settings to a JSON file. Before the state is written to file, the onSave
  /// signal will be emitted.
  void saveToFile(std::string const& fileName) const;

  /// Writes the current settings to a JSON object. Before the state is stored, the onSave
  /// signal will be emitted.
  std::string saveToJson() const;

  // -----------------------------------------------------------------------------------------------

  /// Defines the initial simulation time. Should be either "today" or in the format
  /// "1950-01-02T00:00:00.000Z". When the settings are saved, mStartDate will be set to "today" if
  /// the current simulation time is very similar to the actual system time.
  std::string mStartDate;

  /// When the simulation time is resetted, this date will be used. Should be either "today" or in
  /// the format "1950-01-02 00:00:00.000".
  std::string mResetDate;

  /// Defines the min and max date on the timebar. Changing these values will be directly reflected
  /// in the user interface. Should be in the format "1950-01-02T00:00:00.000Z".
  utils::Property<std::string> pMinDate;
  utils::Property<std::string> pMaxDate;

  /// The current speed of the simulation. A value of zero will cause the simulation to be paused, a
  /// value of one corresponds to real-time. Negative values will cause the time to run backwards.
  utils::DefaultProperty<float> pTimeSpeed{1.F};

  /// All configured CelestialObjects are updated by the SolarSystem each frame. Plugins can draw
  /// geometry at those positions. Each object can have several properties; only "center", "frame",
  /// and "existence" are mandatory. All others can be ommited to use default values. A fully
  /// configured CelestialOject looks like this:
  /// ...,
  /// "objects": {
  ///   "Earth": {
  ///      "center": "Earth",
  ///      "frame": "IAU_Earth",
  ///      "existence": [
  ///        "1950-01-02 00:00:00.000",
  ///        "2049-12-31 00:00:00.000"
  ///      ],
  ///      "position": [0.0, 0.0, 0.0],
  ///      "rotation": [1.0, 0.0, 0.0, 0.0],
  ///      "scale": 1.0,
  ///      "radii": [0.0, 0.0, 0.0],
  ///      "bodyCullingRadius": 0.0,
  ///      "orbitCullingRadius": 0.0,
  ///      "trackable": false,
  ///      "collidable": false
  ///   },
  ///   ...
  /// },
  /// ...
  /// The entries in this map are immutable (const). Therefore, if some attribute of a CelestialBody
  /// needs to be changed, the respective object has to be removed and re-added to the list. Hence
  /// plugins do not have to monitor for changes in each object - they only have to listen for newly
  /// created or removed objects (if they are interested in this).
  cs::utils::ObservableMap<std::string, std::shared_ptr<const cs::scene::CelestialObject>> mObjects;

  /// The values of the observer are updated by the SolarSystem once each frame. For all others,
  /// they should be considered readonly. If you want to modify the transformation of the virtual
  /// observer, use the api of the SolarSystem instead.
  struct Observer {
    /// The name of the SPICE center of the observer.
    utils::Property<std::string> pCenter;

    /// The SPICE frame of reference the observer is currently in.
    utils::Property<std::string> pFrame;

    /// The position of the observer in meters relative to its center and frame.
    utils::Property<glm::dvec3> pPosition;

    /// The rotation of the observer relative to its frame.
    utils::Property<glm::dquat> pRotation;
  } mObserver;

  /// Bookmarks are managed in CosmoScout's core. Plugins can create and delete bookmarks via the
  /// GuiManager's API. A bookmark can have a position in space and / or time. It may also describe
  /// a period in time.
  struct Bookmark {

    /// The location of a bookmark is defined by a SPICE anchor, an optional cartesian position (in
    /// meters) and an optional rotation.
    struct Location {
      std::string               mCenter;
      std::string               mFrame;
      std::optional<glm::dvec3> mPosition;
      std::optional<glm::dquat> mRotation;
    };

    /// The time of a bookmark has an optional end parameter which makes the bookmark describe a
    /// time span rather a time point.
    struct Time {
      std::string                mStart;
      std::optional<std::string> mEnd;
    };

    /// The name of the bookmark is the only required field. It's not strictly required but a good
    /// idea to keep this unique amongst the bookmarks for an anchor.
    std::string mName;

    /// This can be a longer text.
    std::optional<std::string> mDescription;

    /// This can be a name of a png file in share/resources/icons.
    std::optional<std::string> mIcon;

    /// You may use this to visually highlight different types of bookmarks.
    std::optional<glm::vec3> mColor;

    /// Location and Time are both optional, but omitting both results in a pretty useless bookmark.
    std::optional<Location> mLocation;
    std::optional<Time>     mTime;
  };

  /// This list of bookmarks is not updated at runtime. To create new bookmarks and receive updates
  /// on existing bookmarks, use the API of the GuiManager. On settings save, the GuiManager will
  /// update this list of Bookmarks.
  std::vector<Bookmark> mBookmarks;

  /// In order for the scientists to be able to interact with their environment, the next virtual
  /// celestial body must never be more than an arm’s length away.
  /// If the Solar System were always represented on a 1:1 scale, the virtual planetary surface
  /// would be too far away to work effectively with the simulation. The SceneScale object controls
  /// how the virtual scene is scaled depending on the observer's position. This distance to the
  /// closest celestial body depends on the observer's *real* distance in outer space to the
  /// respective body.
  /// If the observer is closer to a celestial body's surface than "closeRealDistance" (in meters),
  /// the scene will be shown in 1:"minScale" and the respective body will be rendered at a distance
  /// of "closeVisualDistance" (in meters). If the observer is farther away than "farRealDistance"
  /// (in meters) from any body, the scene will be shown in 1:"maxScale" and the closest body will
  /// be rendered at a distance of "farVisualDistance" (in meters). At any distance between
  /// "closeRealDistance" and "farRealDistance", the values above will be linearly interpolated.
  /// This object also controls the automatic SPICE frame changes when the observer moves from
  /// body to body. The active body is determined by its weight which is calculated by its size and
  /// distance to the observer. When this weight exceeds "trackWeight", the observer will follow the
  /// body's position. When this weight exceeds "lockWeight", the observer will also follow the
  /// body's rotation.
  struct SceneScale {
    double mMinScale;
    double mMaxScale;
    double mCloseVisualDistance;
    double mFarVisualDistance;
    double mCloseRealDistance;
    double mFarRealDistance;
    double mLockWeight;
    double mTrackWeight;
    double mMinObjectSize;
  };

  SceneScale mSceneScale{};

  // -----------------------------------------------------------------------------------------------

  /// The file name of the meta kernel for SPICE.
  utils::Property<std::string> pSpiceKernel;

  /// If set to false, the user interface is completely hidden.
  utils::DefaultProperty<bool> pEnableUserInterface{true};

  /// If set to true, a ray is shown emerging from your input device.
  utils::DefaultProperty<bool> pEnableMouseRay{false};

  /// If set to true, there will be controls in the user interface to control the camera's
  /// frustum. In a VR setup, this should usually be set to 'false'.
  utils::DefaultProperty<bool> pEnableSensorSizeControl{true};

  /// A list of files which shall be downloaded before the application starts.
  struct DownloadData {
    std::string mUrl;
    std::string mFile;
  };

  std::vector<DownloadData> mDownloadData;

  /// If the (optional) object is given in the configuration file, the user interface is not drawn
  /// in full-screen but rather at the given viewspace postion.
  struct GuiPosition {
    uint32_t mWidthPixel;
    uint32_t mHeightPixel;
    double   mWidthMeter;
    double   mHeightMeter;
    double   mPosXMeter;
    double   mPosYMeter;
    double   mPosZMeter;
    double   mRotX;
    double   mRotY;
    double   mRotZ;
  };

  std::optional<GuiPosition> mGuiPosition;

  /// These set the loglevel for the output to the log file, console and on-screen output
  /// respectively.
  /// trace:    Critical messages, errors, warnings, info, debug and trace messages.
  /// debug:    Critical messages, errors, warnings, info and debug messages.
  /// info:     Critical messages, errors, warnings and info messages.
  /// warning:  Critical messages, errors and warnings.
  /// error:    Critical messages and errors.
  /// critical: Only critical messages.
  /// off:      No output at all.
  utils::DefaultProperty<spdlog::level::level_enum> pLogLevelFile{spdlog::level::debug};
  utils::DefaultProperty<spdlog::level::level_enum> pLogLevelConsole{spdlog::level::trace};
  utils::DefaultProperty<spdlog::level::level_enum> pLogLevelScreen{spdlog::level::info};
  utils::DefaultProperty<spdlog::level::level_enum> pLogLevelGL{spdlog::level::info};

  /// Contains the last 20 commands which have been executed via the on-screen console.
  std::optional<std::deque<std::string>> mCommandHistory;

  // -----------------------------------------------------------------------------------------------

  /// Each body which should cast eclipse shadows needs an eclipse shadow map.
  struct EclipseShadowMap {
    std::optional<std::string> mTexture;
  };

  struct Graphics {
    /// Enables or disables vertical synchronization.
    utils::DefaultProperty<bool> pEnableVsync{true};

    /// A multiplicator for the size of worldspace gui-elements.
    utils::DefaultProperty<double> pWorldUIScale{1.F};

    /// A multiplicator for the size of screenspace gui-elements.
    utils::DefaultProperty<double> pMainUIScale{1.F};

    /// A multiplicator for terrain height.
    utils::DefaultProperty<float> pHeightScale{1.F};

    /// If set to true, HDR rendering will be enabled per default. It can still be disabled at run
    /// time. Defaults to false.
    utils::DefaultProperty<bool> pEnableHDR{false};

    /// If set to false, all shading computations should be disabled.
    utils::DefaultProperty<bool> pEnableLighting{true};

    /// For now, this supports three values (1, 2, 3). Plugins may consider implementing tradeoffs
    /// between performance and quality based on this setting.
    utils::DefaultProperty<int> pLightingQuality{2};

    /// If set to true, shadow maps will be computed.
    utils::DefaultProperty<bool> pEnableShadows{false};

    /// If set to true, plugins may draw some debug information (like cascade coloring) to help
    /// debugging the shadow mapping.
    utils::DefaultProperty<bool> pEnableShadowsDebug{false};

    /// If set to true, the projectionview matrix which is used to calculate the shadow maps is not
    /// updated anymore. This may help debugging of shadow mapping.
    utils::DefaultProperty<bool> pEnableShadowsFreeze{false};

    /// The resolution of each shadow map cascade.
    utils::DefaultProperty<int> pShadowMapResolution{2048};

    /// The number of shadow map cascades.
    utils::DefaultProperty<int> pShadowMapCascades{3};

    /// A parameter to control shadow acne. Hight values lead to a less accurate shadow but also to
    /// less artifacts.
    utils::DefaultProperty<float> pShadowMapBias{1.0F};

    /// The viewspace depth range to compute shadows for.
    utils::DefaultProperty<glm::vec2> pShadowMapRange{glm::vec2(0.F, 100.F)};

    /// The additional size of the shadow frustum in light direction.
    utils::DefaultProperty<glm::vec2> pShadowMapExtension{glm::vec2(-100.F, 100.F)};

    /// An exponent for controlling the distribution of shadow map cascades.
    utils::DefaultProperty<float> pShadowMapSplitDistribution{1.F};

    /// If set to true, the exposure of the virtual camera will be computed automatically. Has no
    /// effect if HDR rendering is disabled.
    utils::DefaultProperty<bool> pEnableAutoExposure{true};

    /// Controls the exposure of the virtual camera. This has no effect if auto exposure is enabled
    /// or if HDR rendering is disabled. Measured in exposure values (EV).
    utils::DefaultProperty<float> pExposure{0.F};

    /// The range from which to choose values for the auto exposure. Measured in exposure values
    /// (EV).
    utils::DefaultProperty<glm::vec2> pAutoExposureRange{glm::vec2(-12.F, 9.F)};

    /// An additional exposure control which is applied after auto exposure. Has no effect if HDR
    /// rendering is disabled. Measured in exposure values (EV).
    utils::DefaultProperty<float> pExposureCompensation{0.F};

    /// An exponent controlling the speed of auto exposure adaption. This has no effect if auto
    /// exposure is enabled or if HDR rendering is disabled.
    utils::DefaultProperty<float> pExposureAdaptionSpeed{3.F};

    /// The size of the virtual camera's sensor. Measured in millimeters.
    utils::DefaultProperty<float> pSensorDiagonal{42.F};

    /// The focal length of the virtual camera. Measured in millimeters.
    utils::DefaultProperty<float> pFocalLength{24.F};

    /// The amount of ambient light. This should be in the range 0-1.
    utils::DefaultProperty<float> pAmbientBrightness{0.F};

    /// The amount of ambient occlusion. This should be in the range 0-1.
    utils::DefaultProperty<float> pAmbientOcclusion{0.5F};

    /// The amount of artifical glare. Has no effect if HDR rendering is disabled. This should be in
    /// the range 0-1. A value of zero disables the glare.
    utils::DefaultProperty<float> pGlareIntensity{0.2F};

    /// Higher values produce a smoother glare.
    utils::DefaultProperty<uint32_t> pGlareQuality{0};

    /// If enabled, the more expensive but much smoother manual bicubic texture filtering is used
    /// for the glare.
    utils::DefaultProperty<bool> pEnableBicubicGlareFilter{true};

    /// Specifies how the glare is computed.
    utils::DefaultProperty<graphics::HDRBuffer::GlareMode> pGlareMode{
        graphics::HDRBuffer::GlareMode::eSymmetricGauss};

    /// Specifies how the tone mapping is computed.
    utils::DefaultProperty<graphics::ToneMappingNode::ToneMappingMode> pToneMappingMode{
        graphics::ToneMappingNode::ToneMappingMode::eFilmic};

    /// This makes illumination calculations assume a fixed sun position in the current SPICE frame.
    /// Using the default value glm::dvec3(0.0) disables this feature.
    utils::DefaultProperty<glm::dvec3> pFixedSunDirection{glm::dvec3(0.0, 0.0, 0.0)};

    /// This maps anchor names to eclipse textures.
    std::optional<std::unordered_map<std::string, EclipseShadowMap>> mEclipseShadowMaps;

    /// The eclipse shadow rendering mode.
    utils::DefaultProperty<EclipseShadowMode> pEclipseShadowMode{EclipseShadowMode::eFastTexture};
  };

  Graphics mGraphics;

  // -----------------------------------------------------------------------------------------------

  /// A map with configuration options for each plugin. The JSON object is not parsed, this is done
  /// by the plugins themselves.
  std::map<std::string, nlohmann::json> mPlugins;

  // -----------------------------------------------------------------------------------------------

  /// As CosmoScout VR is always built together with its plugins, we can ignore this warning.
  CS_WARNINGS_PUSH
  CS_DISABLE_MSVC_WARNING(4275)

  /// An exception that is thrown while parsing the config. Prepends thrown exceptions with a
  /// section name to give the user more detailed information about the root of the error.
  /// The exception can and should be nested.
  class CS_CORE_EXPORT DeserializationException : public std::exception {
   public:
    DeserializationException(std::string property, std::string jsonError);

    const char* what() const noexcept override;

    const std::string mProperty;
    const std::string mJSONError;
    const std::string mMessage;
  };

  CS_WARNINGS_POP

  /// This template is used to retrieve values from json objects. There are two reasons not to
  /// directly use the interface of nlohmann::json: First we want to show more detailed error
  /// messages using the exception type defined above. Second, the deserialize() methods are
  /// overloaded to accept std::optionals and utils::DefaultProperties which behave specially (see
  /// class description at the beginning of this file).
  template <typename T>
  static void deserialize(nlohmann::json const& j, std::string const& property, T& target);

  /// Overload for std::optionals. It will set the given optional to std::nullopt if it does not
  /// exist in the json object.
  template <typename T>
  static void deserialize(
      nlohmann::json const& j, std::string const& property, std::optional<T>& target);

  /// Overload for utils::DefaultProperty. It will set the property to its default state if it does
  /// not exist in the json object.
  template <typename T>
  static void deserialize(
      nlohmann::json const& j, std::string const& property, utils::DefaultProperty<T>& target);

  /// Overload for nlohmann::json. While this seems a bit funny, it is quite useful if you want to
  /// actually save / load json data.
  static void deserialize(
      nlohmann::json const& j, std::string const& property, nlohmann::json& target);

  /// This template is used to set values in json objects. The main reasons not to directly use the
  /// interface of nlohmann::json is that we can overload the serialize() method to accept
  /// std::optionals and utils::DefaultProperties which behave specially (see class description at
  /// the beginning of this file).
  template <typename T>
  static void serialize(nlohmann::json& j, std::string const& property, T const& target);

  /// Overload for std::optionals. It will not write anything if the current value is std::nullopt.
  template <typename T>
  static void serialize(
      nlohmann::json& j, std::string const& property, std::optional<T> const& target);

  /// Overload for utils::DefaultProperty. It will not write anything if the current value is the
  /// default value of the property.
  template <typename T>
  static void serialize(
      nlohmann::json& j, std::string const& property, utils::DefaultProperty<T> const& target);

 private:
  mutable utils::Signal<> mOnLoad;
  mutable utils::Signal<> mOnSave;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

CS_CORE_EXPORT void from_json(nlohmann::json const& j, Settings::GuiPosition& o);
CS_CORE_EXPORT void to_json(nlohmann::json& j, Settings::GuiPosition const& o);
CS_CORE_EXPORT void from_json(nlohmann::json const& j, Settings::Observer& o);
CS_CORE_EXPORT void to_json(nlohmann::json& j, Settings::Observer const& o);
CS_CORE_EXPORT void from_json(nlohmann::json const& j, Settings::Bookmark::Location& o);
CS_CORE_EXPORT void to_json(nlohmann::json& j, Settings::Bookmark::Location const& o);
CS_CORE_EXPORT void from_json(nlohmann::json const& j, Settings::Bookmark::Time& o);
CS_CORE_EXPORT void to_json(nlohmann::json& j, Settings::Bookmark::Time const& o);
CS_CORE_EXPORT void from_json(nlohmann::json const& j, Settings::Bookmark& o);
CS_CORE_EXPORT void to_json(nlohmann::json& j, Settings::Bookmark const& o);
CS_CORE_EXPORT void from_json(nlohmann::json const& j, Settings::DownloadData& o);
CS_CORE_EXPORT void to_json(nlohmann::json& j, Settings::DownloadData const& o);
CS_CORE_EXPORT void from_json(nlohmann::json const& j, Settings::SceneScale& o);
CS_CORE_EXPORT void to_json(nlohmann::json& j, Settings::SceneScale const& o);
CS_CORE_EXPORT void from_json(nlohmann::json const& j, Settings::Graphics& o);
CS_CORE_EXPORT void to_json(nlohmann::json& j, Settings::Graphics const& o);
CS_CORE_EXPORT void from_json(nlohmann::json const& j, Settings& o);
CS_CORE_EXPORT void to_json(nlohmann::json& j, Settings const& o);

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void Settings::deserialize(nlohmann::json const& j, std::string const& property, T& target) {
  try {
    j.at(property).get_to(target);
  } catch (DeserializationException const& e) {
    throw DeserializationException(e.mProperty + " in '" + property + "'", e.mJSONError);
  } catch (std::exception const& e) {
    throw DeserializationException("'" + property + "'", e.what());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void Settings::deserialize(
    nlohmann::json const& j, std::string const& property, std::optional<T>& target) {
  auto iter = j.find(property);
  if (iter != j.end()) {
    try {
      target = iter->get<T>();
    } catch (DeserializationException const& e) {
      throw DeserializationException(e.mProperty + " in '" + property + "'", e.mJSONError);
    } catch (std::exception const& e) {
      throw DeserializationException("'" + property + "'", e.what());
    }
  } else {
    target = std::nullopt;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void Settings::deserialize(
    nlohmann::json const& j, std::string const& property, utils::DefaultProperty<T>& target) {
  auto iter = j.find(property);
  if (iter != j.end()) {
    try {
      target = iter->get<T>();
    } catch (DeserializationException const& e) {
      throw DeserializationException(e.mProperty + " in '" + property + "'", e.mJSONError);
    } catch (std::exception const& e) {
      throw DeserializationException("'" + property + "'", e.what());
    }
  } else {
    target.reset();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void Settings::serialize(nlohmann::json& j, std::string const& property, T const& target) {
  j[property] = target;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void Settings::serialize(
    nlohmann::json& j, std::string const& property, std::optional<T> const& target) {
  if (target) {
    j[property] = target.value();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void Settings::serialize(
    nlohmann::json& j, std::string const& property, utils::DefaultProperty<T> const& target) {
  if (!target.isDefault()) {
    j[property] = target.get();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core

#endif // CS_CORE_SETTINGS_HPP
