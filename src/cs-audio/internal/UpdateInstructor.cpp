////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../logger.hpp"
#include "UpdateInstructor.hpp"
#include "../SourceGroup.hpp"
#include <memory>

namespace cs::audio {

UpdateInstructor::UpdateInstructor() 
  : mSourceUpdateList(std::set<std::weak_ptr<SourceBase>, WeakPtrComparatorSource>())
  , mGroupUpdateList(std::set<std::weak_ptr<SourceGroup>, WeakPtrComparatorGroup>())
  , mAudioControllerUpdate(false) { 
}

UpdateInstructor::~UpdateInstructor() {
  std::cout << "close update instructor" << std::endl;
  mSourceUpdateList.clear();
  mGroupUpdateList.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateInstructor::update(std::shared_ptr<SourceBase> source) {
  mSourceUpdateList.insert(source);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateInstructor::update(std::shared_ptr<SourceGroup> sourceGroup) {
  mGroupUpdateList.insert(sourceGroup);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateInstructor::update(std::shared_ptr<AudioController> audioController) {
  mAudioControllerUpdate = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateInstructor::removeUpdate(std::shared_ptr<SourceBase> source) {
  mSourceUpdateList.erase(source);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateInstructor::removeUpdate(std::shared_ptr<SourceGroup> sourceGroup) {
    mGroupUpdateList.erase(sourceGroup);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateInstructor::removeUpdate(std::shared_ptr<AudioController> audioController) {
  mAudioControllerUpdate = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

UpdateInstructor::UpdateInstruction UpdateInstructor::createUpdateInstruction() {
  UpdateInstruction result;

  if (mAudioControllerUpdate) {
    // update every source and group
    result.updateAll = true;
    result.updateWithGroup = nullptr;
    result.updateSourceOnly = nullptr;

  } else {
    result.updateAll = false;
    result.updateWithGroup = std::make_shared<std::vector<std::shared_ptr<SourceBase>>>();
    result.updateSourceOnly = std::make_shared<std::vector<std::shared_ptr<SourceBase>>>();

    // add group members to updateList
    for (auto groupPtr : mGroupUpdateList) {
      if (groupPtr.expired()) { continue; }
      auto groupMembers = groupPtr.lock()->getMembers();   
      result.updateWithGroup->insert(
        std::end(*(result.updateWithGroup)), std::begin(groupMembers), std::end(groupMembers));
    }

    // Filter out all source that are already part of updateWithGroup and add the rest to 
    // updateSourceOnly. This is done to not run the same source twice through the pipeline.
    for (auto sourcePtr : mSourceUpdateList) {
      if (sourcePtr.expired()) { continue; }
      auto sourceShared = sourcePtr.lock();
      if (std::find(result.updateWithGroup->begin(), result.updateWithGroup->end(), sourceShared)
          == result.updateWithGroup->end()) {
        result.updateSourceOnly->push_back(sourceShared);
      }
    }
  }

  // reset update state
  mSourceUpdateList.clear();
  mGroupUpdateList.clear();
  mAudioControllerUpdate = false;

  return result;
}

} // namespace cs::audio