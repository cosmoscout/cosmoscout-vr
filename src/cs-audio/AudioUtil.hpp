////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_UTILS_HPP
#define CS_AUDIO_UTILS_HPP

#include "../cs-core/Settings.hpp"
#include "cs_audio_export.hpp"
#include <any>
#include <glm/fwd.hpp>
#include <map>
#include <memory>
#include <string>

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
  static double getObserverScaleAt(
      glm::dvec3 position, double ObserverScale, std::shared_ptr<cs::core::Settings> settings);

  /// @brief Prints a settings map. Can, for example, be used on
  /// SourceSettings::getCurrentSettings(), SourceSettings::getUpdateSettings() or
  /// Source::getPlaybackSettings().
  /// @param map map to print
  static void printAudioSettings(std::shared_ptr<std::map<std::string, std::any>> map);
  static void printAudioSettings(const std::shared_ptr<const std::map<std::string, std::any>> map);

  /// @brief Computes a FallOffFactor at which the gain is zero for the given distance
  /// @param distance distance at which the gain of source should be zero
  /// @param model attenuation shape of the distance model
  /// @param fallOffStart fallOffStart of the distance model
  /// @param fallOffEnd fallOffEnd of the distance model
  /// @return fallOffFactor 
  static ALfloat computeFallOffFactor(double distance, std::string model="inverse",
   ALfloat fallOffStart=1.f, ALfloat fallOffEnd=static_cast<ALfloat>(std::numeric_limits<float>::max()));

 private:
};

} // namespace cs::audio

#endif // CS_AUDIO_UTILS_HPP