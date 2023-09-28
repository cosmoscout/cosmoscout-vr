////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_OPEN_AL_MANAGER_HPP
#define CS_AUDIO_OPEN_AL_MANAGER_HPP

#include "Settings.hpp"

#include <AL/al.h>
#include <AL/alc.h>

namespace cs::audio {

class /*CS_AUDIO_EXPORT*/ OpenAlManager {
 public:
  OpenAlManager(std::shared_ptr<core::Settings> settings);

  ~OpenAlManager();
  
  /// Returns a list of all possible Output Devices 
  std::vector<std::string> getDevices();
  bool setDevice(std::string outputDevice);

 private:
  
  bool initOpenAl(std::shared_ptr<core::Settings> settings);

  std::unique_ptr<ALCdevice>  mDevice;
  std::unique_ptr<ALCcontext> mContext;

  // temporary stuff for testing
  bool playTestSound(std::string wavToPlay);
  char* loadWAV(const char* fn, int& chan, int& samplerate, int& bps, int& size, unsigned int& format);
  ALuint sources_temp[1];
  ALuint buffer_temp[1];
  int convertToInt(char* buffer, int len);
  bool isBigEndian();
  // ---------------------------
};

} // namespace cs::audio

#endif // CS_AUDIO_OPEN_AL_MANAGER_HPP
