////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AudioEngine.hpp"
#include "Settings.hpp"

#include "../cs-audio/internal/FileReader.hpp"
#include "../cs-audio/internal/OpenAlManager.hpp"
#include "../cs-audio/Source.hpp"
#include "../cs-audio/SourceSettings.hpp"

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

AudioEngine::AudioEngine(std::shared_ptr<Settings> settings) 
    : mSettings(std::move(settings)) 
    , mOpenAlManager(std::make_unique<audio::OpenAlManager>(mSettings)) 
    , mBufferManager(std::make_shared<audio::BufferManager>()) {

  // Tell the user what's going on.
  logger().debug("Creating AudioEngine.");
  logger().info("OpenAL-Soft Vendor:  {}", alGetString(AL_VENDOR));
  logger().info("OpenAL-Soft Version:  {}", alGetString(AL_VERSION));

  playAmbient("C:/Users/sass_fl/audioCS/audioCSNotes/testFiles/scifi_stereo.wav");
  // playAmbient2();
}

AudioEngine::~AudioEngine() {

std::shared_ptr<audio::Source> AudioEngine::createSource(std::string file, std::shared_ptr<audio::SourceSettings> settings) {
  return std::make_shared<audio::Source>(mBufferManager, file, settings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> AudioEngine::getDevices() {
  if (alcIsExtensionPresent(NULL, "ALC_ENUMERATE_ALL_EXT") == AL_TRUE) {
    logger().info("Available Devices: {}.", alcGetString(nullptr, ALC_ALL_DEVICES_SPECIFIER)); 

  } else if (alcIsExtensionPresent(NULL, "ALC_ENUMERATION_EXT") == AL_TRUE) {
    logger().warn("OpenAL Extensions 'ALC_ENUMERATE_ALL_EXT' not found. Not all available devices might be found!");
    logger().info("Available Devices: {}.", alcGetString(nullptr, ALC_DEVICE_SPECIFIER));

  } else {
    logger().warn("OpenAL Extensions 'ALC_ENUMERATE_ALL_EXT' and 'ALC_ENUMERATION_EXT' not found. Unable to find available devices!");
  }
  return std::vector<std::string>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AudioEngine::setDevice(std::string outputDevice) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioEngine::playAmbient(std::string file) {
  // set Listener
	alListener3i(AL_POSITION, 0, 0, 0);

	ALint listenerOri[] = { 0, 0, 1, 0, 1, 0 };
	alListeneriv(AL_ORIENTATION, listenerOri);

  // set source
  audio::Source source = createSource(file);
  source.play();

  int x, y, z;
  alGetListener3i(AL_POSITION, &x, &y, &z);
  std::cout << "listener Position: " << x << ", " << y << ", " << z << std::endl;
}

void AudioEngine::playAmbient2() {
  alGetError(); // pop error stack

	// set Listener
	alListener3f(AL_POSITION, 0, 0, 0);

	ALint listenerOri[] = { 0, 0, 1, 0, 1, 0 };
	alListeneriv(AL_ORIENTATION, listenerOri);

	// set source
	alGenSources((ALuint)1, sources);

	alSource3i(sources[0], AL_POSITION, 0, 0, 0);

	alSourcei(sources[0], AL_LOOPING, AL_TRUE);

	// set buffer
	alGenBuffers((ALuint)1, buffer);

	unsigned int format;
	int channel, sampleRate, bps, size;
	
	char* data = audio::FileReader::loadWAV("C:/Users/sass_fl/audioCS/audioCSNotes/testFiles/scifi_stereo.wav", channel, sampleRate, bps, size, format);
	if (!data)
		return;

	alBufferData(buffer[0], format, data, size, sampleRate);
	delete[] data;

	// bind buffer to source
	alSourcei(sources[0], AL_BUFFER, buffer[0]);

  alSourcePlay(sources[0]);

  int bufferSize;
  alGetBufferi(buffer[0], AL_SIZE, &bufferSize);
  std::cout << "size: " << bufferSize << std::endl;

}

} // namespace cs::core
