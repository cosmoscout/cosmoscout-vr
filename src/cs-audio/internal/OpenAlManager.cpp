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

// testing
#include <iostream>
#include <fstream>
#include <cstring>

namespace cs::audio {

OpenAlManager::OpenAlManager(std::shared_ptr<core::Settings> settings) {
  if (initOpenAl(settings)) {
    playTestSound("../../../../audioCSNotes/testFiles/exotic_mono.wav");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OpenAlManager::~OpenAlManager() {
  // testing stuff
  alDeleteSources(1, sources_temp);
	alDeleteBuffers(1, buffer_temp);
  // ---------------------

  alcMakeContextCurrent(NULL);
	alcDestroyContext(mContext.get());
	alcCloseDevice(mDevice.get());
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
  // open default device
  mDevice = std::unique_ptr<ALCdevice>(alcOpenDevice(NULL));
  if (!mDevice) {
    return false;
  }

  // create context
  /*
  ALCint attrlist[] = {
    ALC_FREQUENCY, settings->mAudio.pMixerFrequency,
	  ALC_MONO_SOURCES, settings->mAudio.pNumberMonoSources,
	  ALC_STEREO_SOURCES, settings->mAudio.pNumberStereoSources,
	  ALC_REFRESH, settings->mAudio.pRefreshRate,
	  ALC_SYNC, settings->mAudio.pContextSync,
	  ALC_HRTF_SOFT, settings->mAudio.pEnableHRTF
  };
  */
  ALCint attrlist[] = {
    ALC_FREQUENCY, 48000,
	  ALC_MONO_SOURCES, 12,
	  ALC_STEREO_SOURCES, 1,
	  ALC_REFRESH, 30,
	  ALC_SYNC, 1,
	  ALC_HRTF_SOFT, 1
  };

  mContext = std::make_unique<ALCcontext>(alcCreateContext(mDevice.get(), attrlist));
  // mContext = std::unique_ptr<ALCcontext>(alcCreateContext(mDevice.get(), attrlist));
  if (!alcMakeContextCurrent(mContext.get())) {
    return false;
  }

  // check for errors
  if (alcGetError(mDevice.get()) != ALC_NO_ERROR) {
    return false; // TODO
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool OpenAlManager::playTestSound(std::string wavToPlay) {
  alGetError(); // pop error stack

  // set Listener
  alListener3i(AL_POSITION, 0, 0, 0);
  if (alGetError() != AL_NO_ERROR)
    return false;

  // set source
  alGenSources((ALuint)1, sources_temp);
  if (alGetError() != AL_NO_ERROR)
    return false;

  alSource3i(sources_temp[0], AL_POSITION, 0, 0, 0);
  if (alGetError() != AL_NO_ERROR)
    return false;

  alSourcei(sources_temp[0], AL_LOOPING, AL_TRUE);
  if (alGetError() != AL_NO_ERROR)
    return false;

  // set buffer
  alGenBuffers((ALuint)1, buffer_temp);
  if (alGetError() != AL_NO_ERROR)
    return false;

  unsigned int format;
  int channel, sampleRate, bps, size;

  char* data = loadWAV(wavToPlay.c_str(), channel, sampleRate, bps, size, format);
  if (!data)
    return false;

  alBufferData(buffer_temp[0], format, data, size, sampleRate);
  delete[] data;
 
  if (alGetError() != AL_NO_ERROR)
    return false;

  // bind buffer to source
  alSourcei(sources_temp[0], AL_BUFFER, buffer_temp[0]);
  if (alGetError() != AL_NO_ERROR)
    return false;

  // play source
  alSourcePlay(sources_temp[0]);
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

char* OpenAlManager::loadWAV(const char* fn, int& chan, int& samplerate, int& bps, int& size, unsigned int& format)
{
    char fileBuffer[4];
    std::ifstream in(fn, std::ios::binary);
    in.read(fileBuffer, 4);
    if (strncmp(fileBuffer, "RIFF", 4) != 0)
    {
        std::cout << "this is not a valid WAVE file" << std::endl;
        return NULL;
    }
    in.read(fileBuffer, 4);
    in.read(fileBuffer, 4);      //WAVE
    in.read(fileBuffer, 4);      //fmt
    in.read(fileBuffer, 4);      //16
    in.read(fileBuffer, 2);      //1
    in.read(fileBuffer, 2);
    chan = convertToInt(fileBuffer, 2);
    in.read(fileBuffer, 4);
    samplerate = convertToInt(fileBuffer, 4);
    in.read(fileBuffer, 4);
    in.read(fileBuffer, 2);
    in.read(fileBuffer, 2);
    bps = convertToInt(fileBuffer, 2);
    in.read(fileBuffer, 4);      //data
    in.read(fileBuffer, 4);
    size = convertToInt(fileBuffer, 4);
    char* data = new char[size];
    in.read(data, size);

	if (chan == 1)
	{
		if (bps == 8)
		{
			format = AL_FORMAT_MONO8;
		}
		else {
			format = AL_FORMAT_MONO16;
		}
	}
	else {
		if (bps == 8)
		{
			format = AL_FORMAT_STEREO8;
		}
		else {
			format = AL_FORMAT_STEREO16;
		}
	}

    return data;
}

int OpenAlManager::convertToInt(char* buffer, int len)
{
    int a = 0;
    if (!isBigEndian())
        for (int i = 0; i < len; i++)
            ((char*)&a)[i] = buffer[i];
    else
        for (int i = 0; i < len; i++)
            ((char*)&a)[3 - i] = buffer[i];
    return a;
}

bool OpenAlManager::isBigEndian()
{
    int a = 1;
    return !((char*)&a)[0];
}

} // namespace cs::audio
