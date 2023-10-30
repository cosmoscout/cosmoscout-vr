////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SourceGroup.hpp"
#include "Source.hpp"
#include "internal/SettingsMixer.hpp"
#include "internal/UpdateInstructor.hpp"
#include "internal/SourceSettings.hpp"

namespace cs::audio {

SourceGroup::SourceGroup(std::shared_ptr<UpdateInstructor> UpdateInstructor, 
  std::shared_ptr<UpdateConstructor> updateConstructor,
  std::shared_ptr<AudioController> audioController) 
  : SourceSettings(std::move(UpdateInstructor))
  , std::enable_shared_from_this<SourceGroup>()
  , mMembers(std::set<std::shared_ptr<Source>>())
  , mUpdateConstructor(std::move(updateConstructor))
  , mAudioController(std::move(audioController)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SourceGroup::~SourceGroup() {
  reset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::join(std::shared_ptr<Source> source) {
  if (source->mGroup != nullptr) {
    logger().warn("Audio Group Warning: Remove Source form previous group before assigning a new one!");
    return;
  }
  mMembers.insert(source);
  source->mGroup = std::shared_ptr<SourceGroup>(this); // TODO: replace with shared_from_this()

  // apply group settings to newly added source
  if (!mCurrentSettings->empty()) {
    mUpdateConstructor->applyCurrentGroupSettings(source, mAudioController.get(), mCurrentSettings);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::remove(std::shared_ptr<Source> sourceToRemove) {
  if (mMembers.erase(sourceToRemove) == 1) {
    sourceToRemove->mGroup = nullptr;

    // TODO: Remove group setting from sources
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::reset() {
  for (auto sourcePtr : mMembers) {
    sourcePtr->mGroup = nullptr;

    // TODO: Remove group setting from sources
  }
  mMembers.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::set<std::shared_ptr<Source>> SourceGroup::getMembers() const {
  return mMembers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::addToUpdateList() {
  mUpdateInstructor->update(shared_from_this());
}

} // namespace cs::audio