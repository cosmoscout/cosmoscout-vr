////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_AUDIO_SOURCE_GROUP_HPP
#define CS_CORE_AUDIO_SOURCE_GROUP_HPP

#include "cs_audio_export.hpp"
#include "Source.hpp"

#include <memory>
#include <string>
#include <any>
#include <map>
#include <set>

namespace cs::audio {

class CS_AUDIO_EXPORT SourceGroup : public SourceSettings
{
 public:
  explicit SourceGroup();
  ~SourceGroup();

  /// Add a new source to the group
  void add(std::shared_ptr<Source> source);
  /// Remove a source from the group
  void remove(std::shared_ptr<Source> source);
  /// Remove all sources form the group
  void reset();

  friend class AudioController;

 private:
  std::set<std::shared_ptr<Source>> mMemberSources;
};

} // namespace cs::audio

#endif // CS_CORE_AUDIO_SOURCE_GROUP_HPP
