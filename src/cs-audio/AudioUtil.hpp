////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_UTILS_HPP
#define CS_AUDIO_UTILS_HPP

#include "cs_audio_export.hpp"

namespace cs::audio {

class CS_AUDIO_EXPORT AudioUtil {
 public:
  double getObserverScaleAt();

 private:

};

} // namespace cs::audio

#endif // CS_AUDIO_UTILS_HPP