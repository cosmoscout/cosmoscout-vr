////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_SETTINGS_HPP
#define CS_CORE_SETTINGS_HPP

#include "cs_core_export.hpp"

#include "../cs-utils/Property.hpp"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <cstdint>
#include <exception>
#include <glm/glm.hpp>
#include <json.hpp>
#include <map>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace nlohmann {

// A partial template specialization for serialization and deserialization of glm::*vec*.
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

// A partial template specialization for serialization and deserialization of glm::*qua*.
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

// A partial template specialization for cs::utils::Property.
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

namespace cs::core {

/// Most of CosmoScout VR's configuration is done with one huge JSON file. This contains some global
/// options and settings for each plugin. The available global options are defined below, the
/// per-plugin settings are defined in each and every plugin.
struct CS_CORE_EXPORT Settings {

  template <typename T>
  struct Default {
    Default(T const& defaultValue)
        : mDefaultValue(defaultValue)
        , mValue(defaultValue) {
    }

    bool isDefault() const {
      return mValue == mDefaultValue;
    }

    void reset() {
      mValue = mDefaultValue;
    }

    T const& operator()() {
      return mValue;
    }

    T& operator->() {
      return mValue;
    }

    T const& operator->() const {
      return mValue;
    }

    void operator=(T const& value) {
      mValue = value;
    }

    const T mDefaultValue;
    T       mValue;
  };

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

  struct Location {
    std::string mPlanet;
    std::string mPlace;
  };

  struct Event {
    std::string                mStart;
    std::optional<std::string> mEnd;
    std::string                mContent;
    std::string                mId;
    std::optional<std::string> mStyle;
    std::string                mDescription;
    std::optional<Location>    mLocation;
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

  /// Defines the min and max Date on the timebar
  std::string mMinDate;
  std::string mMaxDate;

  /// Defines the initial observer location.
  Observer mObserver;

  /// Parameters which define how the virtual scene is scaled based on the observer position.
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

  /// When set to true, HDR rendering will be enabled per default. It can still be disabled at run
  /// time. Defaults to false.
  std::optional<bool> mEnableHDR;

  /// When set to true, a ray is shown emerging from your input device.
  bool mEnableMouseRay;

  /// When set to true, there will be controls in the user interface to control the camera's
  /// frustum. In a VR setup, this should usually be set to 'false'.
  bool mEnableSensorSizeControl;

  /// These set the loglevel for the output to the log file, console and on-screen output
  /// respectively.
  /// trace:    Critical messages, errors, warnings, info, debug and trace messages.
  /// debug:    Critical messages, errors, warnings, info and debug messages.
  /// info:     Critical messages, errors, warnings and info messages.
  /// warning:  Critical messages, errors and warnings.
  /// error:    Critical messages and errors.
  /// critical: Only critical messages.
  /// off:      No output at all.
  spdlog::level::level_enum mFileLogLevel;
  spdlog::level::level_enum mConsoleLogLevel;
  spdlog::level::level_enum mScreenLogLevel;

  /// In order to reduce duplication of code, a list of all used SPICE-frames ("Anchors") is
  /// required at the start of each configuration file. The name of each Anchor is then later used
  /// to reference the respective SPICE frame.
  std::map<std::string, Anchor> mAnchors;

  /// Events to show on the timenavigation bar
  std::vector<Event> mEvents;

  /// A map with configuration options for each plugin. The JSON object is not parsed, this is done
  /// by the plugins themselves.
  std::map<std::string, nlohmann::json> mPlugins;

  /// Initializes all members from a given JSON file.
  void read(std::string const& fileName);

  /// Writes the current settings to a JSON file.
  void write(std::string const& fileName) const;

  /// An exception that is thrown while parsing the config. Prepends thrown exceptions with a
  /// section name to give the user more detailed information about the root of the error.
  /// The exception can and should be nested.
  class DeserializationException : public std::exception {
   public:
    DeserializationException(std::string const& property, std::string const& jsonError);

    const char* what() const noexcept override;

    const std::string mProperty;
    const std::string mJSONError;
    const std::string mMessage;
  };

  /// This template is used to retrieve values from json objects. There are two reasons not to
  /// directly use the interface of nlohmann::json: First we want to show more detailed error
  /// messages using the exception type defined above. Second, the and deserialize() methods are
  /// overloaded to accept std::optionals.
  /// This makes using optional paameters in your settings very straight-forward.
  template <typename T>
  static void deserialize(nlohmann::json const& j, std::string const& property, T& target) {
    try {
      j.at(property).get_to(target);
    } catch (DeserializationException const& e) {
      throw DeserializationException(e.mProperty + " in '" + property + "'", e.mJSONError);
    } catch (std::exception const& e) {
      throw DeserializationException("'" + property + "'", e.what());
    }
  }

  /// Overload for std::optionals. It will set the given optional to std::nullopt if it does not
  /// exist in the json object.
  template <typename T>
  static void deserialize(
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

  template <typename T>
  static void deserialize(
      nlohmann::json const& j, std::string const& property, Default<T>& target) {
    try {
      target = j.at(property).get<T>();
    } catch (DeserializationException const& e) {
      throw DeserializationException(e.mProperty + " in '" + property + "'", e.mJSONError);
    } catch (std::exception const& e) {
      throw DeserializationException("'" + property + "'", e.what());
    }
  }

  /// This template is used to set values in json objects. The main reasons not to directly use the
  /// interface of nlohmann::json is that we can overload the serialize() method to accept
  /// std::optionals.
  /// This makes using optional paameters in your settings very straight-forward.
  template <typename T>
  static void serialize(nlohmann::json& j, std::string const& property, T const& target) {
    j[property] = target;
  }

  template <typename T>
  static void serialize(
      nlohmann::json& j, std::string const& property, std::optional<T> const& target) {
    if (target) {
      j[property] = target.value();
    }
  }

  template <typename T>
  static void serialize(nlohmann::json& j, std::string const& property, Default<T> const& target) {
    if (!target.isDefault()) {
      j[property] = target.mValue;
    }
  }
};

std::pair<double, double> CS_CORE_EXPORT getExistenceFromSettings(
    std::pair<std::string, Settings::Anchor> const& anchor);

} // namespace cs::core

#endif // CS_CORE_SETTINGS_HPP
