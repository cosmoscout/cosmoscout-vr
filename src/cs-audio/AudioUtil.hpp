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
  static double getObserverScaleAt(glm::dvec3 position, double ObserverScale, 
    std::shared_ptr<cs::core::Settings> settings);

  static void printAudioSettings(
    std::shared_ptr<std::map<std::string, std::any>> map);
  static void printAudioSettings(
    const std::shared_ptr<const std::map<std::string, std::any>> map);

 private:

};

} // namespace cs::audio

#endif // CS_AUDIO_UTILS_HPP