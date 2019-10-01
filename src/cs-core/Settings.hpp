////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_SETTINGS_HPP
#define CS_CORE_SETTINGS_HPP

#include "cs_core_export.hpp"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <cstdint>
#include <exception>
#include <glm/glm.hpp>
#include <json.hpp>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace nlohmann {

/// A partial template specialisation for serialization and deserialization of std::optional.
template <typename T>
struct adl_serializer<std::optional<T>> {
  static void to_json(json& j, const std::optional<T>& opt) {
    if (!opt) {
      j = nullptr;
    } else {
      // This will call adl_serializer<T>::to_json which will find the free function to_json in T's
      // namespace!
      j = *opt;
    }
  }

  static void from_json(const json& j, std::optional<T>& opt) {
    if (j.is_null()) {
      opt = {};
    } else {
      // Same as above, but with adl_serializer<T>::from_json.
      opt = j.get<T>();
    }
  }
};

} // namespace nlohmann

namespace cs::core {

/// Most of CosmoScout VR's configuration is done with one huge JSON file. This contains some global
/// options and settings for each plugin. The available global options are defined below, the
/// per-plugin settings are defined in each and every plugin.
class CS_CORE_EXPORT Settings {
 public:
  struct Anchor {
    std::string mCenter;
    std::string mFrame;
    std::string mStartExistence;
    std::string mEndExistence;
  };

  struct Gui {
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

  struct Observer {
    std::string mCenter;
    std::string mFrame;
    double      mLongitude;
    double      mLatitude;
    double      mDistance;
  };

  struct DownloadData {
    std::string mUrl;
    std::string mFile;
  };

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
    double mNearClip;
    double mMinFarClip;
    double mMaxFarClip;
  };

  /// Defines the initial simulation time.
  std::string mStartDate;

  /// Defines the initial observer location.
  Observer mObserver;

  /// PArameters which define how the virtual scene is scaled based on the observer position.
  SceneScale mSceneScale;

  /// A list of files which shall be downloaded before the application starts.
  std::vector<DownloadData> mDownloadData;

  /// The file name of the meta kernel for SPICE.
  std::string mSpiceKernel;

  /// When the (optional) object is given in the configuration file, the user interface is not drawn
  /// in full-screen but rather at the given viewspace postion.
  std::optional<Gui> mGui;

  /// A multiplicator for the size of worldspace gui-elements.
  float mWidgetScale;

  /// When set to true, a ray is shown emerging from your input device.
  bool mEnableMouseRay;

  /// In order to reduce duplication of code, a list of all used SPICE-frames ("Anchors") is
  /// required at the start of each configuration file. The name of each Anchor is then later used
  /// to reference the respective SPICE frame.
  std::map<std::string, Anchor> mAnchors;

  /// A map with configuration options for each plugin. The JSON object is not parsed, this is done
  /// by the plugins themselves.
  std::map<std::string, nlohmann::json> mPlugins;

  /// Creates an instance of this struct from a given JSON file.
  static Settings read(std::string const& fileName);
};

std::pair<double, double> CS_CORE_EXPORT getExistenceFromSettings(
    std::pair<std::string, Settings::Anchor> const& anchor);

/// An exception that is thrown while parsing the config. Prepends thrown exceptions with a section
/// name to give the user more detailed information about the root of the error.
/// The exception can and should be nested.
/// @see parseSection()
class CS_CORE_EXPORT SettingsSectionException : public std::exception {
  const std::string completeMessage;

 public:
  const std::string sectionName;
  const std::string message;

  SettingsSectionException(std::string sectionName, std::string message)
      : sectionName(std::move(sectionName))
      , message(std::move(message))
      , completeMessage(
            "Failed to parse settings config in section '" + sectionName + "': " + message){};

  [[nodiscard]] const char* what() const noexcept override {
    return completeMessage.c_str();
  }
};

/// Parses a section of the config file. If an exception gets thrown inside f a new exception will
/// be created with the sectionName in its error message.
///
/// If this method is nested the section names of all the method calls will be joined with a 'dot'.
/// Example:
///
/// @code
/// parseSection("foo", [] {
///     parseSection("bar", [] {
///         throw std::runtime_error("bad character");
///     });
/// });
/// @endcode
///
/// Console output: Failed to parse settings config in section 'foo.bar': bad character
void CS_CORE_EXPORT parseSection(std::string const& sectionName, std::function<void()> const& f);

/// A settings section that is also a value.
template <typename T>
T parseSection(std::string const& sectionName, nlohmann::json const& j) {
  T result;
  parseSection(sectionName, [&] { result = j.at(sectionName).get<T>(); });
  return result;
}

/// An optional settings section.
template <typename T>
std::optional<T> parseOptionalSection(std::string const& sectionName, nlohmann::json const& j) {
  std::optional<T> result;

  auto iter = j.find(sectionName);
  if (iter != j.end()) {
    cs::core::parseSection(sectionName, [&] { result = iter->get<std::optional<T>>(); });
  }

  return result;
}

/// A map of key values.
template <typename K, typename V>
std::map<K, V> parseMap(std::string const& sectionName, nlohmann::json const& j) {
  return parseSection<std::map<K, V>>(sectionName, j);
}

/// A vector of settings.
template <typename T>
std::vector<T> parseVector(std::string const& sectionName, nlohmann::json const& j) {
  return parseSection<std::vector<T>>(sectionName, j);
}

/// A single atomic property.
template <typename T>
T parseProperty(std::string const& propertyName, nlohmann::json const& j) {
  try {
    return j.at(propertyName).get<T>();
  } catch (std::exception const& e) {
    throw std::runtime_error(
        "Error while trying to parse property '" + propertyName + "': " + std::string(e.what()));
  }
}

/// An optional property.
template <typename T>
std::optional<T> parseOptional(std::string const& propertyName, nlohmann::json const& j) {
  auto iter = j.find(propertyName);
  if (iter != j.end()) {
    return iter->get<std::optional<T>>();
  } else {
    return std::nullopt;
  }
}

} // namespace cs::core

#endif // CS_CORE_SETTINGS_HPP
