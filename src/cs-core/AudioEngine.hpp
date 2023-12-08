////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_AUDIO_AUDIOENGINE_HPP
#define CS_CORE_AUDIO_AUDIOENGINE_HPP

#include "GuiManager.hpp"
#include "Settings.hpp"
#include "cs_audio_export.hpp"

#include "../cs-audio/AudioController.hpp"
#include "../cs-audio/Source.hpp"
#include "../cs-audio/SourceGroup.hpp"
#include "../cs-audio/StreamingSource.hpp"
#include "../cs-audio/internal/BufferManager.hpp"
#include "../cs-audio/internal/OpenAlManager.hpp"
#include "../cs-audio/internal/ProcessingStepsManager.hpp"
#include "../cs-audio/internal/UpdateConstructor.hpp"
#include "../cs-utils/Property.hpp"

#include <any>
#include <map>

namespace cs::core {

/// @brief The AudioEngine is responsible for initializing all necessary audio classes.
/// It also provides access to create audio controllers. This class should only be instantiated
/// once. This instance will be passed to all plugins.
class CS_CORE_EXPORT AudioEngine : public std::enable_shared_from_this<AudioEngine> {
 public:
  AudioEngine(const AudioEngine& obj) = delete;
  AudioEngine(AudioEngine&&)          = delete;

  AudioEngine& operator=(const AudioEngine&) = delete;
  AudioEngine& operator=(AudioEngine&&) = delete;

  AudioEngine(std::shared_ptr<Settings> settings, std::shared_ptr<GuiManager> guiManager);
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
  std::shared_ptr<core::Settings>                    mSettings;
  std::shared_ptr<audio::OpenAlManager>              mOpenAlManager;
  std::shared_ptr<audio::BufferManager>              mBufferManager;
  std::shared_ptr<audio::ProcessingStepsManager>     mProcessingStepsManager;
  std::shared_ptr<core::GuiManager>                  mGuiManager;
  std::vector<std::weak_ptr<audio::AudioController>> mAudioControllers;
  std::shared_ptr<audio::UpdateConstructor>          mUpdateConstructor;
  utils::Property<float>                             mMasterVolume;
  bool                                               mIsLeader;

  /// Creates the Audio GUI Settings
  void createGUI();
};

} // namespace cs::core

#endif // CS_CORE_AUDIO_AUDIOENGINE_HPP
