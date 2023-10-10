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

namespace cs::audio {

struct CS_AUDIO_EXPORT SourceGroup {
 public:
  void add(std::shared_ptr<Source> source);
  void update();
  std::unique_ptr<SourceSettings>      mSettings;

 private:
  std::vector<std::shared_ptr<Source>> mSources;
};

} // namespace cs::audio

#endif // CS_CORE_AUDIO_SOURCE_GROUP_HPP
