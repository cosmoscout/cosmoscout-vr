////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SourceGroup.hpp"
#include "internal/SettingsMixer.hpp"
#include "internal/SourceBase.hpp"
#include "internal/SourceSettings.hpp"
#include "internal/UpdateInstructor.hpp"
#include "logger.hpp"

namespace cs::audio {

SourceGroup::SourceGroup(std::shared_ptr<UpdateInstructor> UpdateInstructor,
    std::shared_ptr<UpdateConstructor>                     updateConstructor,
    std::shared_ptr<AudioController>                       audioController)
    : SourceSettings(std::move(UpdateInstructor))
    , std::enable_shared_from_this<SourceGroup>()
    , mMembers(std::set<std::weak_ptr<SourceBase>, WeakPtrComparatorSource>())
    , mUpdateConstructor(std::move(updateConstructor))
    , mAudioController(audioController) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SourceGroup::SourceGroup()
    : SourceSettings(false)
    , std::enable_shared_from_this<SourceGroup>() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SourceGroup::~SourceGroup() {
  if (mIsLeader) {
    reset();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::join(std::shared_ptr<SourceBase> source) {
  if (!mIsLeader) {
    return;
  }
  if (mAudioController.expired()) {
    logger().warn(
        "Group warning: AudioController of group is expired! Unable to assign source to group!");
    return;
  }

  auto currentGroup = source->getGroup();
  if (currentGroup != shared_from_this()) {
    source->setGroup(shared_from_this());

    mMembers.insert(source);

    // apply group settings to newly added source
    if (!mCurrentSettings->empty()) {
      mUpdateConstructor->applyCurrentGroupSettings(
          source, mAudioController.lock(), mCurrentSettings);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::leave(std::shared_ptr<SourceBase> sourceToRemove) {
  if (!mIsLeader) {
    return;
  }
  if (mMembers.erase(sourceToRemove) == 1) {
    sourceToRemove->leaveGroup();

    if (mAudioController.expired()) {
      logger().warn("Group warning: AudioController of group is expired! Unable remove group "
                    "settings from source!");
      return;
    }
    mUpdateConstructor->removeCurrentGroupSettings(sourceToRemove, mAudioController.lock());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::reset() {
  if (!mIsLeader) {
    return;
  }
  for (auto sourcePtr : mMembers) {
    if (sourcePtr.expired()) {
      continue;
    }
    sourcePtr.lock()->leaveGroup();

    if (mAudioController.expired()) {
      logger().warn("Group warning: AudioController of group is expired! Unable remove group "
                    "settings from source!");
      continue;
    }
    mUpdateConstructor->removeCurrentGroupSettings(sourcePtr.lock(), mAudioController.lock());
  }
  mMembers.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::vector<std::shared_ptr<SourceBase>> SourceGroup::getMembers() {
  if (!mIsLeader) {
    std::vector<std::shared_ptr<SourceBase>>();
  }
  std::vector<std::shared_ptr<SourceBase>> membersShared(mMembers.size());
  for (auto member : mMembers) {

    if (member.expired()) {
      mMembers.erase(member);
      continue;
    }

    membersShared.emplace_back(member.lock());
  }
  return membersShared;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::addToUpdateList() {
  mUpdateInstructor->update(shared_from_this());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::removeFromUpdateList() {
  mUpdateInstructor->removeUpdate(shared_from_this());
}

} // namespace cs::audio