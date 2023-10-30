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

SourceGroup::SourceGroup(std::shared_ptr<UpdateInstructor> UpdateInstructor) 
  : SourceSettings(UpdateInstructor)
  , std::enable_shared_from_this<SourceGroup>()
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SourceGroup::~SourceGroup() {
  reset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::add(std::shared_ptr<Source> source) {
  if (source->mGroup != nullptr) {
    logger().warn("Audio Group Warning: Remove Source form previous group before assigning a new one!");
    return;
  }
  mMemberSources.insert(source);
  source->mGroup = std::shared_ptr<SourceGroup>(this);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::remove(std::shared_ptr<Source> sourceToRemove) {
  // if removal was successful
  if (mMemberSources.erase(sourceToRemove) == 1) {
    sourceToRemove->mGroup = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::reset() {
  for (auto sourcePtr : mMemberSources) {
    sourcePtr->mGroup = nullptr;
  }
  mMemberSources.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::set<std::shared_ptr<Source>> SourceGroup::getMembers() const {
  return mMemberSources;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::addToUpdateList() {
  mUpdateInstructor->update(shared_from_this());
}

} // namespace cs::audio