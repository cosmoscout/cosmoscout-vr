////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_OPEN_AL_MANAGER_HPP
#define CS_AUDIO_OPEN_AL_MANAGER_HPP

#include "cs_audio_export.hpp"
#include "../../cs-core/Settings.hpp"
#include "../../cs-utils/Property.hpp"

#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alext.h>

namespace cs::audio {

class CS_AUDIO_EXPORT OpenAlManager {
 public:
  OpenAlManager(const OpenAlManager& obj) = delete;
  OpenAlManager(OpenAlManager&&) = delete;

  OpenAlManager& operator=(const OpenAlManager&) = delete;
  OpenAlManager& operator=(OpenAlManager&&) = delete;

  static std::shared_ptr<OpenAlManager> createOpenAlManager();
  ~OpenAlManager();

  bool initOpenAl(core::Settings::Audio settings);
  /// Returns a list of all possible Output Devices 
  std::vector<std::string> getDevices();
  /// Sets the output device for the audioEngine
  bool setDevice(std::string outputDevice);

 private:
  ALCdevice*          mDevice;
  ALCcontext*         mContext;
  std::vector<ALCint> mAttributeList;

  OpenAlManager();
  bool contextErrorOccurd();

  // OpenALSoft extensions function pointers:
  LPALCREOPENDEVICESOFT alcReopenDeviceSOFT;
};

} // namespace cs::audio

#endif // CS_AUDIO_OPEN_AL_MANAGER_HPP
