////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_OPEN_AL_MANAGER_HPP
#define CS_AUDIO_OPEN_AL_MANAGER_HPP

#include "../../cs-core/Settings.hpp"
#include "cs_audio_export.hpp"

#include <AL/alc.h>
#include <AL/alext.h>

namespace cs::audio {

/// @brief This class handles the initialization (Device and Context) of OpenAL and the
/// functionality of getting and setting an audio output device. This class should only be
/// instantiated once.
class CS_AUDIO_EXPORT OpenAlManager {
 public:
  OpenAlManager(const OpenAlManager& obj) = delete;
  OpenAlManager(OpenAlManager&&)          = delete;

  OpenAlManager& operator=(const OpenAlManager&) = delete;
  OpenAlManager& operator=(OpenAlManager&&) = delete;

  OpenAlManager();
  ~OpenAlManager();

  /// @brief Initializes OpenAL by opening a device and creating a context.
  /// @return True if successful
  bool initOpenAl(core::Settings::Audio settings);

  /// @brief Checks for all available output devices. Either by the ALC_ENUMERATE_ALL_EXT extension
  /// or, if not available, the ALC_ENUMERATE_EXT extension.
  /// @return List of name of all available devices
  std::vector<std::string> getDevices();

  /// @brief Try's to set the provided device name as the OpenAL output device via the
  /// alcReopenDeviceSOFT extension.
  /// @return True if successful
  bool setDevice(std::string outputDevice);

 private:
  /// Pointer to the current device
  ALCdevice* mDevice;
  /// Pointer to the current context
  ALCcontext* mContext;
  /// Specifies the current settings for OpenAL. The attributes are set via the config file.
  std::vector<ALCint> mAttributeList;

  /// @brief Checks if an OpenAL Context Error occurred and if so prints a logger warning containing
  /// the error.
  /// @return True if an error occurred
  bool contextErrorOccurd();

  // OpenALSoft extensions function pointers:
  LPALCREOPENDEVICESOFT alcReopenDeviceSOFT;
};

} // namespace cs::audio

#endif // CS_AUDIO_OPEN_AL_MANAGER_HPP
