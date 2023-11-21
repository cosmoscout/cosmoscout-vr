////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../logger.hpp"
#include "UpdateInstructor.hpp"
#include "../SourceGroup.hpp"

namespace cs::audio {

UpdateInstructor::UpdateInstructor() 
  : mSourceUpdateList(std::set<std::shared_ptr<SourceBase>>())
  , mGroupUpdateList(std::set<std::shared_ptr<SourceGroup>>())
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

  } else {
    result.updateAll = false;
    result.updateWithGroup = std::make_shared<std::vector<std::shared_ptr<SourceBase>>>();
    result.updateSourceOnly = std::make_shared<std::vector<std::shared_ptr<SourceBase>>>();

    // add group members to updateList
    for (auto groupPtr : mGroupUpdateList) {
      auto groupMembers = groupPtr->getMembers();   
      result.updateWithGroup->insert(std::end(*(result.updateWithGroup)), std::begin(groupMembers), std::end(groupMembers));
    }

    // Filter out all source that are already part of updateWithGroup and add the rest to updateSourceOnly. This is done to not run the 
    // same source twice through the pipeline.
    for (auto sourcePtr : mSourceUpdateList) {
      if (std::find(result.updateWithGroup->begin(), result.updateWithGroup->end(), sourcePtr) == result.updateWithGroup->end()) {
        result.updateSourceOnly->push_back(sourcePtr);
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