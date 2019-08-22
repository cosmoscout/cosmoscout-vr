////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_SETTINGS_HPP
#define CS_CORE_SETTINGS_HPP

#include "cs_core_export.hpp"

#include <cstdint>
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

  struct Event {
    std::string mStart;
    std::optional<std::string> mEnd;
    std::string mContent;
    std::string mId;
    std::optional<std::string> mStyle;
  };

  /// Defines the initial simulation time.
  std::string mStartDate;

  /// Defines the min and max Date on the timebar
  std::string mMinDate;
  std::string mMaxDate;

  /// Defines the initial observer location.
  Observer mObserver;

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

  /// Events to show on the timenavigation bar
  std::vector<Event> mEvents;

  /// A map with configuration options for each plugin. The JSON object is not parsed, this is done
  /// by the plugins themselves.
  std::map<std::string, nlohmann::json> mPlugins;

  /// Creates an instance of this struct from a given JSON file.
  static Settings read(std::string const& fileName);
};

} // namespace cs::core

#endif // CS_CORE_SETTINGS_HPP
