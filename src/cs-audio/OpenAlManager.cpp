////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "OpenAlManager.hpp"
#include "Settings.hpp"
#include <memory>

namespace cs::audio {

OpenAlManager::OpenAlManager(std::shared_ptr<core::Settings::audio> settings) {
  
  initOpenAl(settings);
}

bool OpenAlManager::initOpenAl(std::shared_ptr<core::Settings::audio> settings) {
  // open default device
  mDevice = std::unique_ptr<ALCdevice>(alcOpenDevice(NULL));
  if (!mDevice) {
    return false;
  }

  // create context
  ALCint attrlist[] = {
    ALC_FREQUENCY, settings->pMixerOutputFrequency,
	ALC_MONO_SOURCES, settings->pNumberMonoSources,
	ALC_STEREO_SOURCES, settings->pNumberStereoSources,
	ALC_REFRESH, settings->pRefreshRate,
	ALC_SYNC, (settings->pContextSync ? AL_TRUE : AL_FALSE),
	ALC_HRTF_SOFT, (settings->pEnableHRTF ? AL_TRUE : AL_FALSE)
  };

  mContext = std::unique_ptr<ALCcontext>(alcCreateContext(mDevice.get(), attrlist));
  if (!alcMakeContextCurrent(mContext.get())) {
    return false;
  }

  // check for errors
  if (alcGetError(mDevice.get()) != ALC_NO_ERROR) {
    return false; // TODO
  }
}

} // namespace cs::graphics
