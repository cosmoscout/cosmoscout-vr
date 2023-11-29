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

SourceBase::~SourceBase() {
  std::cout << "close SourceBase" << std::endl;
  alGetError(); // clear error code
  alDeleteSources(1, &mOpenAlId);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to delete source!");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceBase::play() {
  set("playback", std::string("play"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceBase::stop() {
  set("playback", std::string("stop"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceBase::pause() {
  set("playback", std::string("pause"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string SourceBase::getFile() const {
  return mFile;   
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const ALuint SourceBase::getOpenAlId() const {
  return mOpenAlId;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::shared_ptr<std::map<std::string, std::any>> SourceBase::getPlaybackSettings() const {
  return mPlaybackSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceBase::addToUpdateList() {
  mUpdateInstructor->update(shared_from_this());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceBase::removeFromUpdateList() {
  mUpdateInstructor->removeUpdate(shared_from_this());  
}

const std::shared_ptr<SourceGroup> SourceBase::getGroup() {
  if (mGroup.expired()) {
    return nullptr;
  }
  return mGroup.lock();
}

void SourceBase::setGroup(std::shared_ptr<SourceGroup> newGroup) {
  leaveGroup();
  mGroup = newGroup;
  newGroup->join(shared_from_this());
}

void SourceBase::leaveGroup() {
  if (!mGroup.expired()) {
    auto sharedGroup = mGroup.lock();
    mGroup.reset();
    sharedGroup->remove(shared_from_this());
  }
}

} // namespace cs::audio