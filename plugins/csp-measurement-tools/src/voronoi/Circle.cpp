////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Circle.hpp"
#include "Arc.hpp"
#include "Breakpoint.hpp"

#include <cmath>
#include <limits>

namespace csp::measurementtools {

Circle::Circle(Arc* a, double sweepLine)
    : mSite(a->mSite)
    , mArc(a)
    , mIsValid(true) {

  if (!mArc->mLeftBreak || !mArc->mRightBreak) {
    mIsValid = false;
    return;
  }

  Site* site3(&mArc->mLeftBreak->mLeftArc->mSite);
  Site* site2(&mArc->mSite);
  Site* site1(&mArc->mRightBreak->mRightArc->mSite);

  if ((site2->mX - site1->mX) * (site3->mY - site1->mY) -
          (site3->mX - site1->mX) * (site2->mY - site1->mY) >
      0) {
    mIsValid = false;
    return;
  }

  // Algorithm from O'Rourke 2ed p. 189.
  const double A = site2->mX - site1->mX;
  const double B = site2->mY - site1->mY;
  const double C = site3->mX - site1->mX;
  const double D = site3->mY - site1->mY;
  const double E = A * (site1->mX + site2->mX) + B * (site1->mY + site2->mY);
  const double F = C * (site1->mX + site3->mX) + D * (site1->mY + site3->mY);
  const double G = 2 * (A * (site3->mY - site2->mY) - B * (site3->mX - site2->mX));

  // Points are co-linear.
  if (std::fabs(G) <= std::numeric_limits<double>::epsilon()) {
    mIsValid = false;
    return;
  }

  mCenter.mX = (D * E - B * F) / G;
  mCenter.mY = (A * F - C * E) / G;

  mPriority =
      Vector2f(mCenter.mX, mCenter.mY + (mCenter - Vector2f(site1->mX, site1->mY)).length());

  if (mPriority.mY < sweepLine) {
    mIsValid = false;
    return;
  }

  mArc->mEvent = this;
}

bool operator<(Circle const& lhs, Circle const& rhs) {
  return (lhs.mPriority.mY == rhs.mPriority.mY) ? (lhs.mPriority.mX < rhs.mPriority.mX)
                                                : (lhs.mPriority.mY > rhs.mPriority.mY);
}
} // namespace csp::measurementtools
