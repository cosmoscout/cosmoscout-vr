////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SourceBase.hpp"
#include "BufferManager.hpp"
#include "alErrorHandling.hpp"
#include "SettingsMixer.hpp"
#include "../logger.hpp"
#include "../SourceGroup.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

SourceBase::SourceBase(std::string file,
  std::shared_ptr<UpdateInstructor> UpdateInstructor)
  : SourceSettings(UpdateInstructor) 
  , std::enable_shared_from_this<SourceBase>()
  , mFile(std::move(file)) 
  , mPlaybackSettings(std::make_shared<std::map<std::string, std::any>>()) {
  
  alGetError(); // clear error code

  // generate new source  
  alGenSources((ALuint)1, &mOpenAlId);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to generate OpenAL-Soft Source!");
    return;
  }

  // positions needs to be set relative in case the listener moves:
  // set position to listener relative
  alSourcei(mOpenAlId, AL_SOURCE_RELATIVE, AL_TRUE);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source position specification to relative!");
    return;
  }

  alSource3i(mOpenAlId, AL_POSITION, 0, 0, 0);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source position to (0, 0, 0)!");
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////

SourceBase::SourceBase()
  : SourceSettings(false) 
  , std::enable_shared_from_this<SourceBase>() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SourceBase::~SourceBase() {
  if (mIsLeader) {
    alGetError(); // clear error code
    alDeleteSources(1, &mOpenAlId);
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to delete source!");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceBase::play() {
  if (!mIsLeader) { return; }
  set("playback", std::string("play"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceBase::stop() {
  if (!mIsLeader) { return; }
  set("playback", std::string("stop"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceBase::pause() {
  if (!mIsLeader) { return; }
  set("playback", std::string("pause"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string SourceBase::getFile() const {
  if (!mIsLeader) { return std::string(); }
  return mFile;   
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const ALuint SourceBase::getOpenAlId() const {
  if (!mIsLeader) { return 0; }
  return mOpenAlId;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::shared_ptr<const std::map<std::string, std::any>> SourceBase::getPlaybackSettings() const {
  if (!mIsLeader) { std::shared_ptr<const std::map<std::string, std::any>>(); }
  return mPlaybackSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceBase::addToUpdateList() {
  if (!mIsLeader) { return; }
  mUpdateInstructor->update(shared_from_this());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceBase::removeFromUpdateList() {
  if (!mIsLeader) { return; }
  mUpdateInstructor->removeUpdate(shared_from_this());
}

const std::shared_ptr<SourceGroup> SourceBase::getGroup() {
  if (!mIsLeader) { std::shared_ptr<SourceGroup>(); }
  if (mGroup.expired()) {
    return nullptr;
  }
  return mGroup.lock();
}

void SourceBase::setGroup(std::shared_ptr<SourceGroup> newGroup) {
  if (!mIsLeader) { return; }
  leaveGroup();
  mGroup = newGroup;
  newGroup->join(shared_from_this());
}

void SourceBase::leaveGroup() {
  if (!mIsLeader) { return; }
  if (!mGroup.expired()) {
    auto sharedGroup = mGroup.lock();
    mGroup.reset();
    sharedGroup->leave(shared_from_this());
  }
}

} // namespace cs::audio