////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Beachline.hpp"

#include "Arc.hpp"
#include "VoronoiGenerator.hpp"

namespace csp::measurementtools {

Beachline::Beachline(VoronoiGenerator* parent)
    : mParent(parent)
    , mRoot(nullptr) {
}

Arc* Beachline::insertArcFor(Site const& site) {
  // if site creates the very first Arc of the Beachline
  if (mRoot == nullptr) {
    mRoot = new Arc(site); // NOLINT(cppcoreguidelines-owning-memory): TODO
    return mRoot;
  }

  Arc* newArc = new Arc(site); // NOLINT(cppcoreguidelines-owning-memory): TODO

  Arc* brokenArcLeft = mBreakPoints.empty() ? mRoot : mBreakPoints.getArcAt(site.mX);
  brokenArcLeft->invalidateEvent();

  // site inserted at exactly the same height as brokenArcLeft
  if (site.mY == brokenArcLeft->mSite.mY) {
    if (site.mX < brokenArcLeft->mSite.mX) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): TODO
      newArc->mRightBreak = new Breakpoint(newArc, brokenArcLeft, mParent);
      mParent->addTriangulationEdge(brokenArcLeft->mSite, newArc->mSite);
      brokenArcLeft->mLeftBreak = newArc->mRightBreak;
      mBreakPoints.insert(newArc->mRightBreak);
    }
    // new one is right of brokenArcLeft
    else {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): TODO
      newArc->mLeftBreak = new Breakpoint(brokenArcLeft, newArc, mParent);
      mParent->addTriangulationEdge(brokenArcLeft->mSite, newArc->mSite);
      brokenArcLeft->mRightBreak = newArc->mLeftBreak;
      mBreakPoints.insert(newArc->mLeftBreak);
    }
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): TODO
    Arc* brokenArcRight = new Arc(brokenArcLeft->mSite);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): TODO
    newArc->mLeftBreak = new Breakpoint(brokenArcLeft, newArc, mParent);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): TODO
    newArc->mRightBreak = new Breakpoint(newArc, brokenArcRight, mParent);

    mParent->addTriangulationEdge(brokenArcLeft->mSite, newArc->mSite);

    brokenArcRight->mRightBreak = brokenArcLeft->mRightBreak;
    if (brokenArcRight->mRightBreak) {
      brokenArcRight->mRightBreak->mRightArc->mLeftBreak->mLeftArc = brokenArcRight;
    }
    brokenArcRight->mLeftBreak = newArc->mRightBreak;
    brokenArcLeft->mRightBreak = newArc->mLeftBreak;

    mBreakPoints.insert(newArc->mLeftBreak);
    mBreakPoints.insert(newArc->mRightBreak);
  }

  return newArc;
}

void Beachline::removeArc(Arc* arc) {
  Arc* leftArc  = arc->mLeftBreak ? arc->mLeftBreak->mLeftArc : nullptr;
  Arc* rightArc = arc->mRightBreak ? arc->mRightBreak->mRightArc : nullptr;

  arc->invalidateEvent();
  if (leftArc) {
    leftArc->invalidateEvent();
  }
  if (rightArc) {
    rightArc->invalidateEvent();
  }

  if (leftArc && rightArc) {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): TODO
    auto* merged = new Breakpoint(leftArc, rightArc, mParent);

    mParent->addTriangulationEdge(leftArc->mSite, rightArc->mSite);

    leftArc->mRightBreak = merged;
    rightArc->mLeftBreak = merged;

    mBreakPoints.remove(arc->mRightBreak);
    mBreakPoints.remove(arc->mLeftBreak);

    mBreakPoints.insert(merged);

    delete arc->mLeftBreak;  // NOLINT(cppcoreguidelines-owning-memory): TODO
    delete arc->mRightBreak; // NOLINT(cppcoreguidelines-owning-memory): TODO
  } else if (leftArc) {
    mBreakPoints.remove(arc->mLeftBreak);
    leftArc->mRightBreak = nullptr;
    delete arc->mLeftBreak; // NOLINT(cppcoreguidelines-owning-memory): TODO
  } else if (rightArc) {
    mBreakPoints.remove(arc->mRightBreak);
    rightArc->mLeftBreak = nullptr;
    delete arc->mRightBreak; // NOLINT(cppcoreguidelines-owning-memory): TODO
  }

  delete arc; // NOLINT(cppcoreguidelines-owning-memory): TODO
}

void Beachline::finish(std::vector<Edge>& edges) {
  mBreakPoints.finishAll(edges);
}
} // namespace csp::measurementtools
