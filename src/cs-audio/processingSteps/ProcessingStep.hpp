////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PROCESSING_STEP_HPP
#define CS_AUDIO_PROCESSING_STEP_HPP

#include "cs_audio_export.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

class CS_AUDIO_EXPORT ProcessingStep {
 public:
  
  /// Every derived class of ProcessingStep must implement a static create() function.
  /// Defining it here is not possible as virtual static function are not possible in C++.
  /// An alternative would be to use the Curiously Recurring Template Pattern (CRTP) but this approach would 
  /// require an additional abstract parent class because with CRTP the ProcessingStep class would become
  /// a template class which prevents the storage of all derived classes inside a single container.
  // virtual static std::shared_ptr<ProcessingStep> create() = 0;

  /// @brief processes a source with the given settings
  /// @param source Source to process
  /// @param settings settings to apply
  /// @param failedSettings Pointer to list which contains all failed settings
  virtual void process(std::shared_ptr<Source> source, 
    std::shared_ptr<std::map<std::string, std::any>> settings,
    std::shared_ptr<std::vector<std::string>> failedSettings) = 0;

  /// @return Wether the processing requires an update call each frame
  virtual bool requiresUpdate() const = 0;

  /// @brief update function to call each frame
  virtual void update() = 0;
};

} // namespace cs::audio

#endif // CS_AUDIO_PROCESSING_STEP_HPP
