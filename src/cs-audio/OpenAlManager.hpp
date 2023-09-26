////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_OPEN_AL_MANAGER_HPP
#define CS_AUDIO_OPEN_AL_MANAGER_HPP

#include "Settings.hpp"

namespace cs::audio {

class /*CS_GRAPHICS_EXPORT*/ OpenAlManager {
 public:
  OpenAlManager(std::shared_ptr<core::Settings::audio> settings);

  ~OpenAlManager();
  
  /// Returns a list of all possible Output Devices 
  std::vector<std::string> getDevices();
  bool setDevice(std::string outputDevice);

 private:
  
  bool initOpenAl(std::shared_ptr<core::Settings::audio> settings);

  std::unique_ptr<ALCdevice>  mDevice;
  std::unique_ptr<ALCcontext> mContext;
};

} // namespace cs::audio

#endif // CS_AUDIO_OPEN_AL_MANAGER_HPP
