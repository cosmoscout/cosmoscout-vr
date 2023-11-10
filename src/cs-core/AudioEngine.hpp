////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_AUDIO_AUDIOENGINE_HPP
#define CS_CORE_AUDIO_AUDIOENGINE_HPP

#include "cs_audio_export.hpp"
#include "Settings.hpp"
#include "SolarSystem.hpp"
#include "GuiManager.hpp"

#include "../cs-audio/internal/OpenAlManager.hpp"
#include "../cs-audio/Source.hpp"
#include "../cs-audio/StreamingSource.hpp"
#include "../cs-audio/SourceGroup.hpp"
#include "../cs-audio/AudioController.hpp"
#include "../cs-audio/internal/BufferManager.hpp"
#include "../cs-audio/internal/ProcessingStepsManager.hpp"
#include "../cs-audio/internal/UpdateConstructor.hpp"
#include "../cs-utils/Property.hpp"

#include <map>
#include <any>

namespace cs::core {

class CS_CORE_EXPORT AudioEngine {

 public:
  AudioEngine(const AudioEngine& obj) = delete;
  AudioEngine(AudioEngine&&) = delete;

  AudioEngine& operator=(const AudioEngine&) = delete;
  AudioEngine& operator=(AudioEngine&&) = delete;

  static std::shared_ptr<AudioEngine> createAudioEngine(std::shared_ptr<Settings> settings, 
  std::shared_ptr<SolarSystem> solarSystem, std::shared_ptr<GuiManager> guiManager);
  ~AudioEngine();

  /// @brief Returns a list of all possible Output Devices (wrapper to the OpenAlManager function)
  std::vector<std::string> getDevices();
  
  /// @brief Sets the output device for the audioEngine (wrapper to the OpenAlManager function)
  bool setDevice(std::string outputDevice);
  
  /// @brief Sets the master volume for the audioEngine 
  /// @return Whether it was successful
  bool setMasterVolume(float gain);
  
  /// @brief Update function to call every frame. Currently calls only the update function of the 
  /// processing steps manager. 
  void update();
  
  /// @brief Create a new AudioController
  /// @return Ptr to the new controller
  std::shared_ptr<audio::AudioController> createAudioController();

 private:
  std::shared_ptr<core::Settings>                      mSettings;
  std::shared_ptr<audio::OpenAlManager>                mOpenAlManager;
  std::shared_ptr<audio::BufferManager>                mBufferManager;
  std::shared_ptr<audio::ProcessingStepsManager>       mProcessingStepsManager;
  std::shared_ptr<core::GuiManager>                    mGuiManager;
  utils::Property<float>                               mMasterVolume;
  std::vector<std::shared_ptr<audio::AudioController>> mAudioControllers;
  std::shared_ptr<audio::UpdateConstructor>            mUpdateConstructor;

  AudioEngine(std::shared_ptr<Settings> settings, 
    std::shared_ptr<SolarSystem> solarSystem, std::shared_ptr<GuiManager> guiManager);

  /// Creates the Audio GUI Settings
  void createGUI();

  // for testing
  std::shared_ptr<audio::AudioController> controllerAmbient;
  std::shared_ptr<audio::AudioController> controllerSpace;
  void playAmbient();
  std::shared_ptr<audio::Source> testSourceAmbient;
  std::shared_ptr<audio::StreamingSource> testSourceStreaming;
  std::shared_ptr<audio::Source> testSourcePosition1;
  std::shared_ptr<audio::Source> testSourcePosition2;
  std::shared_ptr<audio::SourceGroup> testSourceGroup;
  std::map<std::string, std::any> testSettings;

  cs::scene::CelestialObserver mObserver; // ?
  std::shared_ptr<SolarSystem> mSolarSystem; // ?
};

} // namespace cs::core

#endif // CS_CORE_AUDIO_AUDIOENGINE_HPP
