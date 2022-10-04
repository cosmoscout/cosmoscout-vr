////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Arc.hpp"
#include "Breakpoint.hpp"

namespace csp::measurementtools {

Arc::Arc(Site const& s)
    : mSite(s)
    , mLeftBreak(nullptr)
    , mRightBreak(nullptr)
    , mEvent(nullptr) {
}

void Arc::invalidateEvent() {

  if (mEvent) {
    if (mEvent->mIsValid) {
      mEvent->mIsValid = false;
    }

    mEvent = nullptr;
  }
}
} // namespace csp::measurementtools
