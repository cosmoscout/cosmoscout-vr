////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "OpenAlManager.hpp"
#include "../../cs-core/Settings.hpp"
#include <memory>

#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alext.h>

namespace cs::audio {

std::shared_ptr<OpenAlManager> OpenAlManager::createOpenAlManager() {
  auto static openAlManager = std::shared_ptr<OpenAlManager>(new OpenAlManager());
  return openAlManager;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OpenAlManager::OpenAlManager() {}

////////////////////////////////////////////////////////////////////////////////////////////////////

OpenAlManager::~OpenAlManager() {
  alcMakeContextCurrent(nullptr);
	alcDestroyContext(mContext);
	alcCloseDevice(mDevice);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool OpenAlManager::initOpenAl(core::Settings::Audio settings) {
  // create settings for context
  mAttributeList[0]  = ALC_FREQUENCY;
  mAttributeList[1]  = settings.pMixerFrequency.get();
  mAttributeList[2]  = ALC_MONO_SOURCES;
  mAttributeList[3]  = settings.pNumberMonoSources.get();
  mAttributeList[4]  = ALC_STEREO_SOURCES;
  mAttributeList[5]  = settings.pNumberStereoSources.get();
  mAttributeList[6]  = ALC_REFRESH;
  mAttributeList[7]  = settings.pRefreshRate.get();
  mAttributeList[8]  = ALC_SYNC;
  mAttributeList[9]  = settings.pContextSync.get();
  mAttributeList[10] = ALC_HRTF_SOFT;
  mAttributeList[11] = settings.pEnableHRTF.get();

  // open default device
  mDevice = alcOpenDevice(nullptr);
  if (!mDevice) {
    logger().warn("Failed to open default device!");
    return false;
  }

  // create context
  mContext = alcCreateContext(mDevice, mAttributeList.data());
  if (contextErrorOccurd()) {
    logger().warn("Failed to create context!");
    return false;
  }

  // select context
  ALCboolean contextSelected = alcMakeContextCurrent(mContext);
  if (contextErrorOccurd() || !contextSelected) {
    logger().warn("Faild to select current context!");
    return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool OpenAlManager::setDevice(std::string outputDevice) {
  if (alcIsExtensionPresent(NULL, "ALC_SOFT_reopen_device") == ALC_FALSE) {
    logger().warn("OpenAL Extensions 'ALC_SOFT_reopen_device' not found. Unable to change the output device!");
    return false;
  }

  if (alcReopenDeviceSOFT == nullptr) {
    alcReopenDeviceSOFT = (LPALCREOPENDEVICESOFT)alGetProcAddress("alcReopenDeviceSOFT");
  }

  if (alcReopenDeviceSOFT(mDevice, outputDevice.c_str(), mAttributeList.data()) == ALC_FALSE) { // TODO: Fails sometimes?
    contextErrorOccurd();
    logger().warn("Failed to set the new output device! Playback remains on the current device!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> OpenAlManager::getDevices() {
  std::vector<std::string> result;
  int macro;

  if (alcIsExtensionPresent(NULL, "ALC_ENUMERATE_ALL_EXT") == ALC_TRUE) {
    macro = ALC_ALL_DEVICES_SPECIFIER;
  
  } else if (alcIsExtensionPresent(NULL, "ALC_ENUMERATION_EXT") == ALC_TRUE) {
    logger().warn("OpenAL Extensions 'ALC_ENUMERATE_ALL_EXT' not found. Not all available devices might be found!");
    macro = ALC_DEVICE_SPECIFIER;

  } else {
    logger().warn("OpenAL Extensions 'ALC_ENUMERATE_ALL_EXT' and 'ALC_ENUMERATION_EXT' not found. Unable to find available devices!");
    return result;
  }

  const ALCchar* device = alcGetString(nullptr, macro);
  const ALCchar* next = alcGetString(nullptr, macro) + 1;
  size_t len = 0;

  while (device && *device != '\0' && next && *next != '\0') {
    result.push_back(device);
    len = strlen(device);
    device += (len + 1);
    next += (len + 2);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool OpenAlManager::contextErrorOccurd() {
  ALCenum error;
  if ((error = alcGetError(mDevice)) != ALC_NO_ERROR) {

    std::string errorCode;
    switch(error) {
      case ALC_INVALID_DEVICE:
        errorCode = "Invalid device handle";
        break;
      case ALC_INVALID_CONTEXT:
        errorCode = "Invalid context handle";
        break;
      case ALC_INVALID_ENUM:
        errorCode = "Invalid enumeration passed to an ALC call";
        break;
      case ALC_INVALID_VALUE:
        errorCode = "Invalid value passed to an ALC call";
        break;
      case ALC_OUT_OF_MEMORY:
        errorCode = "Not enough memory to execute the ALC call";
        break;
      default:
        errorCode = "Unkown error code";
    }
    logger().warn("OpenAL-Soft Context Error occured! Reason: {}...", errorCode);
    return true;
  }
  return false;
}
} // namespace cs::audio
