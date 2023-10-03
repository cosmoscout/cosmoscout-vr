////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_OPEN_AL_MANAGER_HPP
#define CS_AUDIO_OPEN_AL_MANAGER_HPP

#include "cs_audio_export.hpp"
#include "Settings.hpp"

#include <AL/al.h>
#include <AL/alc.h>

namespace cs::audio {

class CS_AUDIO_EXPORT OpenAlManager {
 public:
  OpenAlManager(std::shared_ptr<core::Settings> settings);
  ~OpenAlManager();

 private:
  // using raw pointers because smart one always throw an error
  ALCdevice* mDevice;
  ALCcontext* mContext;
  
  bool initOpenAl(std::shared_ptr<core::Settings> settings);
  bool contextErrorOccurd();
};

} // namespace cs::audio

#endif // CS_AUDIO_OPEN_AL_MANAGER_HPP
