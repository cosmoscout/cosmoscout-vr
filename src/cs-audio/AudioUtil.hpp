////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_UTILS_HPP
#define CS_AUDIO_UTILS_HPP

#include "cs_audio_export.hpp"
#include "../cs-core/Settings.hpp"
#include <memory>
#include <map>
#include <string>
#include <any>
#include <glm/fwd.hpp>

namespace cs::audio {

class CS_AUDIO_EXPORT AudioUtil {
 public:
  /// @brief Computes the scale the observer would have if he would be at the provided position.
  /// This can be useful as a starting point to scale a source, either as a spatialized sphere or 
  /// it's fallOffStart distance. If the source is far away from the next celestial object, meaning 
  /// the observer scale is very large, it can be very hard to navigate in such a way that the
  /// source is audible because even the smallest change of postion can lead to a very large change
  /// of the real world position.  
  /// @param position Position at which to compute the observer scale
  /// @param ObserverScale Scale at the current Observer Position
  /// @param settings settings
  /// @return Observer scale at position
  static double getObserverScaleAt(glm::dvec3 position, double ObserverScale, 
    std::shared_ptr<cs::core::Settings> settings);

  /// @brief Prints a settings map. Can, for example, be used on SourceSettings::getCurrentSettings(),
  /// SourceSettings::getUpdateSettings() or Source::getPlaybackSettings().
  /// @param map map to print
  static void printAudioSettings(
    std::shared_ptr<std::map<std::string, std::any>> map);
  static void printAudioSettings(
    const std::shared_ptr<const std::map<std::string, std::any>> map);

 private:

};

} // namespace cs::audio

#endif // CS_AUDIO_UTILS_HPP