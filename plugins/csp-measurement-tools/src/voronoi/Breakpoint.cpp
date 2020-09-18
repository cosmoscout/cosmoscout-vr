////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Breakpoint.hpp"
#include "Arc.hpp"
#include "VoronoiGenerator.hpp"

#include <cmath>

namespace csp::measurementtools {

Breakpoint::Breakpoint() = default;

Breakpoint::Breakpoint(Arc* left, Arc* right, VoronoiGenerator* generator)
    : mLeftArc(left)
    , mRightArc(right)
    , mLeftChild(nullptr)
    , mRightChild(nullptr)
    , mParent(nullptr)
    , mGenerator(generator)
    , mStart(position()) {
}

Vector2f const& Breakpoint::position() const {
  double currentSweepLine = mGenerator->sweepLine();

  if (mSweepline == currentSweepLine) {
    return mPosition;
  }

  mSweepline = currentSweepLine;
  updatePosition();
  return mPosition;
}

Edge Breakpoint::finishEdge(Vector2f const& end) const {
  return std::make_pair(mStart, end);
}

void Breakpoint::updatePosition() const {

  double pX = mLeftArc->mSite.mX;
  double pY = mLeftArc->mSite.mY;
  double rX = mRightArc->mSite.mX;
  double rY = mRightArc->mSite.mY;

  if (pY == rY) {
    mPosition.mX = (pX + rX) * 0.5;
  } else if (rY == mSweepline) {
    mPosition.mX = rX;
  } else if (pY == mSweepline) {
    mPosition.mX = pX;
    pX           = rX;
    pY           = rY;
  } else {
    double leftDiff  = 2 * (pY - mSweepline);
    double rightDiff = 2 * (rY - mSweepline);

    // Use the quadratic formula.
    double a = 1.0 / leftDiff - 1.0 / rightDiff;
    double b = -2.0 * (pX / leftDiff - rX / rightDiff);
    double c = (pX * pX + pY * pY - mSweepline * mSweepline) / leftDiff -
               (rX * rX + rY * rY - mSweepline * mSweepline) / rightDiff;
    mPosition.mX = (-b - std::sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
  }

  // Plug back into one of the parabola equations.
  if (pY != mSweepline) {
    mPosition.mY = (pY * pY + (pX - mPosition.mX) * (pX - mPosition.mX) - mSweepline * mSweepline) /
                   (2.0 * pY - 2.0 * mSweepline);
  } else {
    mPosition.mY = mGenerator->minY();
  }
}
} // namespace csp::measurementtools
