////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_TYPES_HPP
#define CSP_VISUAL_QUERY_TYPES_HPP

#include <string>
#include <vector>
#include <any>
#include "GL/glew.h"

namespace csp::visualquery {

struct Bound {
  double min;
  double max;
};

struct Dimension {
  int mWidth;
  int mLength;
  int mDepth;
};

struct TimeStamp {
  double mTimeStamp; // time in TDB
};

template <typename T>
struct Values {
  std::vector<T> values;
  GLenum         type;
};

class Image2D {
 public:
  std::any  mValues;
  Bound     mBoundX;
  Bound     mBoundY;
  TimeStamp mTimeStamp;
  Dimension mDimension;
};

class LayeredImage2D {
 public:
  std::vector<std::any> mValues;
  Bound                 mBoundX;
  Bound                 mBoundY;
  TimeStamp             mTimeStamp;
  Dimension             mDimension;
};

class Volume3D {
 public:
  std::any  mValues;
  Bound     mBoundX;
  Bound     mBoundY;
  Bound     mBoundZ;
  TimeStamp mTimeStamp;
  Dimension mDimension;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_TYPES_HPP