////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BreakpointTree.hpp"
#include "Arc.hpp"
#include "Breakpoint.hpp"

#include <cmath>
#include <limits>

namespace csp::measurementtools {

BreakpointTree::BreakpointTree() = default;

BreakpointTree::~BreakpointTree() {
  clear(mRoot);
}

void BreakpointTree::insert(Breakpoint* point) {
  if (empty()) {
    mRoot = point;
  } else {
    insert(point, mRoot);
  }
}

void BreakpointTree::remove(Breakpoint* point) {
  if (point->mParent == nullptr) {

    if (point->mLeftChild && point->mRightChild) {
      mRoot          = point->mRightChild;
      mRoot->mParent = nullptr;
      attachLeftOf(point->mLeftChild, mRoot);
    } else if (point->mLeftChild) {
      mRoot          = point->mLeftChild;
      mRoot->mParent = nullptr;
    } else if (point->mRightChild) {
      mRoot          = point->mRightChild;
      mRoot->mParent = nullptr;
    } else {
      mRoot = nullptr;
    }
  } else {

    const bool isLeftChild(point == point->mParent->mLeftChild);

    if (point->mLeftChild && point->mRightChild) {
      if (isLeftChild) {
        point->mParent->mLeftChild = point->mLeftChild;
        point->mLeftChild->mParent = point->mParent;
        attachRightOf(point->mRightChild, point->mParent->mLeftChild);
      } else {
        point->mParent->mRightChild = point->mRightChild;
        point->mRightChild->mParent = point->mParent;
        attachLeftOf(point->mLeftChild, point->mParent->mRightChild);
      }
    } else if (point->mLeftChild) {
      if (isLeftChild) {
        point->mParent->mLeftChild = point->mLeftChild;
      } else {
        point->mParent->mRightChild = point->mLeftChild;
      }
      point->mLeftChild->mParent = point->mParent;
    } else if (point->mRightChild) {
      if (isLeftChild) {
        point->mParent->mLeftChild = point->mRightChild;
      } else {
        point->mParent->mRightChild = point->mRightChild;
      }
      point->mRightChild->mParent = point->mParent;
    } else {
      if (isLeftChild) {
        point->mParent->mLeftChild = nullptr;
      } else {
        point->mParent->mRightChild = nullptr;
      }
    }
  }
}

Arc* BreakpointTree::getArcAt(double x) const {

  Breakpoint* nearest = getNearestNode(x, mRoot);

  if (x < nearest->position().mX) {
    return nearest->mLeftArc;
  }

  return nearest->mRightArc;
}

bool BreakpointTree::empty() const {
  return !mRoot;
}

void BreakpointTree::insert(Breakpoint* newNode, Breakpoint* atNode) {
  double newX = newNode->position().mX;
  double atX  = atNode->position().mX;
  if (newX < atX || (newX == atX && newNode->mRightArc == atNode->mLeftArc)) {
    if (atNode->mLeftChild) {
      insert(newNode, atNode->mLeftChild);
    } else {
      atNode->mLeftChild = newNode;
      newNode->mParent   = atNode;
    }
  } else {
    if (atNode->mRightChild) {
      insert(newNode, atNode->mRightChild);
    } else {
      atNode->mRightChild = newNode;
      newNode->mParent    = atNode;
    }
  }
}

Breakpoint* BreakpointTree::getNearestNode(double x, Breakpoint* current) const {
  if (!current) {
    return nullptr;
  }

  Breakpoint* nearestChild = (x < current->position().mX) ? getNearestNode(x, current->mLeftChild)
                                                          : getNearestNode(x, current->mRightChild);
  Breakpoint* nearest = current;

  if (nearestChild &&
      (std::fabs(x - nearestChild->position().mX) < std::fabs(x - nearest->position().mX))) {
    nearest = nearestChild;
  }

  return nearest;
}

void BreakpointTree::finishAll(std::vector<Edge>& edges) {
  finishAll(edges, mRoot);
}

void BreakpointTree::finishAll(std::vector<Edge>& edges, Breakpoint* atNode) {
  if (atNode) {
    edges.push_back(atNode->finishEdge(atNode->position()));
    finishAll(edges, atNode->mLeftChild);
    finishAll(edges, atNode->mRightChild);
  }
}

void BreakpointTree::clear(Breakpoint* atNode) {
  if (atNode) {
    clear(atNode->mLeftChild);
    clear(atNode->mRightChild);

    if (atNode->mLeftArc) {
      if (atNode->mLeftArc->mLeftBreak) {
        atNode->mLeftArc->mLeftBreak->mRightArc = nullptr;
      }
      delete atNode->mLeftArc; // NOLINT(cppcoreguidelines-owning-memory): TODO
    }

    if (atNode->mRightArc) {
      if (atNode->mRightArc->mRightBreak) {
        atNode->mRightArc->mRightBreak->mLeftArc = nullptr;
      }
      delete atNode->mRightArc; // NOLINT(cppcoreguidelines-owning-memory): TODO
    }

    delete atNode; // NOLINT(cppcoreguidelines-owning-memory): TODO
  }
}

void BreakpointTree::attachRightOf(Breakpoint* newNode, Breakpoint* atNode) {
  if (atNode->mRightChild) {
    attachRightOf(newNode, atNode->mRightChild);
  } else {
    atNode->mRightChild = newNode;
    newNode->mParent    = atNode;
  }
}

void BreakpointTree::attachLeftOf(Breakpoint* newNode, Breakpoint* atNode) {
  if (atNode->mLeftChild) {
    attachLeftOf(newNode, atNode->mLeftChild);
  } else {
    atNode->mLeftChild = newNode;
    newNode->mParent   = atNode;
  }
}
} // namespace csp::measurementtools
