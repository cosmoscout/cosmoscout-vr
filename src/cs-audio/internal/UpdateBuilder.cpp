////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "UpdateBuilder.hpp"

namespace cs::audio {

UpdateBuilder::UpdateBuilder() 
  : mSourceUpdateList(std::set<std::shared_ptr<Source>>())
  , mGroupUpdateList(std::set<std::shared_ptr<SourceGroup>>())
  , mPluginUpdate(false) { 
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateBuilder::update(Source* source) {
  mSourceUpdateList.insert(std::shared_ptr<Source>(source));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateBuilder::update(SourceGroup* sourceGroup) {
  mGroupUpdateList.insert(std::shared_ptr<SourceGroup>(sourceGroup));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateBuilder::updatePlugin() {
  mPluginUpdate = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

UpdateBuilder::UpdateList UpdateBuilder::createUpdateList() {
  UpdateList result;

  if (mPluginUpdate) {
    // update every source and group
    result.updateAll = true;
    return result;
  }

  // add group members to updateList
  for (auto groupPtr : mGroupUpdateList) {
    auto groupMembers = groupPtr->getMembers();   
    result.updateWithGroup.insert(std::end(result.updateWithGroup), std::begin(groupMembers), std::end(groupMembers));
  }

  // Compute mSourceUpdateList without result.updateWithGroup in order to later only process sources 
  // that are not already in the group update.
  for (auto sourcePtr : mSourceUpdateList) {
    if (std::find(result.updateWithGroup.begin(), result.updateWithGroup.end(), sourcePtr) == result.updateWithGroup.end()) {
      result.updateOnlySource.push_back(sourcePtr);
    }
  }

  // reset update state
  mSourceUpdateList.clear();
  mGroupUpdateList.clear();
  mPluginUpdate = false;

  return result;
}

} // namespace cs::audio