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

OpenAlManager::OpenAlManager(std::shared_ptr<core::Settings> settings) {
  initOpenAl(settings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OpenAlManager::~OpenAlManager() {
  alcMakeContextCurrent(nullptr);
	alcDestroyContext(mContext);
	alcCloseDevice(mDevice);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> OpenAlManager::getDevices() {
  return std::vector<std::string>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool OpenAlManager::setDevice(std::string outputDevice) {
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool OpenAlManager::initOpenAl(std::shared_ptr<core::Settings> settings) {
  // create settings for context
  ALCint attrlist[] = {
    ALC_FREQUENCY, settings->mAudio.pMixerFrequency.get(),
	  ALC_MONO_SOURCES, settings->mAudio.pNumberMonoSources.get(),
	  ALC_STEREO_SOURCES, settings->mAudio.pNumberStereoSources.get(),
	  ALC_REFRESH, settings->mAudio.pRefreshRate.get(),
	  ALC_SYNC, settings->mAudio.pContextSync.get(),
	  ALC_HRTF_SOFT, settings->mAudio.pEnableHRTF.get()
  };

  // open default device
  mDevice = alcOpenDevice(nullptr);
  if (!mDevice) {
    return false;
  }

  // create context
  mContext = alcCreateContext(mDevice, attrlist);
  if (!alcMakeContextCurrent(mContext)) {
    return false;
  }

  // check for errors
  if (alcGetError(mDevice) != ALC_NO_ERROR) {
    return false; // TODO
  }
  return true;
}

} // namespace cs::audio
