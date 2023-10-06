////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SourceGroup.hpp"
#include "Source.hpp"

namespace cs::audio {

void SourceGroup::add(std::shared_ptr<Source> source) {
  mSources.push_back(source);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::update() {
  for (std::shared_ptr<Source> source : mSources) {

  }
}

} // namespace cs::audio