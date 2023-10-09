////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "types.hpp"

#include <vector>
#include <ctime>
#include <optional>

namespace csp::visualquery {

TimeStamp::TimeStamp(double timeStamp)
  : mTimeStamp(timeStamp) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double TimeStamp::getTimeStamp() {
    return mTimeStamp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeStamp::setTimeStamp(double timeStamp) {
    mTimeStamp = timeStamp;
}

//Dimension/////////////////////////////////////////////////////////////////////////////////////////

Dimension::Dimension(int width, int length, int depth)
  : mWidth(width)
  , mLength(length)
  , mDepth(depth) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int Dimension::getDimension(std::string dimensionType) {
  if (dimensionType == "width") {
    return mWidth;
  }
  if (dimensionType == "length") {
    return mLength;
  }
  if (dimensionType == "depth") {
    return mDepth;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Dimension::setDimension(int width, int length, int depth) {
  mWidth = width;
  mLength = length;
  mDepth = depth;
}

void Dimension::setDimension(std::string dimensionType, int value) {
  if (dimensionType == "width") {
    mWidth = value;
    return;
  }
  if (dimensionType == "length") {
    mLength = value;
    return;
  }
  if (dimensionType == "depth") {
    mDepth = value;
    return;
  }
}

//Image2D///////////////////////////////////////////////////////////////////////////////////////////

Image2D::Image2D(std::vector<Point2D> points, double timeStamp, Bound boundX, Bound boundY, Dimension dimension) 
 : mTimeStamp(timeStamp)
 , mPoints(points)
 , mBoundX(boundX)
 , mBoundY(boundY)
 , mDimension(dimension) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Image2D::setPoints(std::vector<Point2D> points) {
  mPoints = points;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Image2D::setBound(std::string boundType, float min, float max) {
  if (boundType == "x") {
    mBoundX = Bound{min, max};

  } else if (boundType == "y") {
    mBoundY = Bound{min, max};
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<Point2D> Image2D::getPoints() {
  return mPoints;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<Bound> Image2D::getBound(std::string boundType) {
  if (boundType == "x") {
    return mBoundX;

  } else if (boundType == "y") {
    return mBoundY;
  }
}

//LayeredImage2D////////////////////////////////////////////////////////////////////////////////////

LayeredImage2D::LayeredImage2D(std::vector<std::vector<Point2D>> points, double timeStamp, Bound boundX, 
  Bound boundY, Dimension dimension) 
 : mTimeStamp(timeStamp)
 , mPoints(points)
 , mBoundX(boundX)
 , mBoundY(boundY)
 , mDimension(dimension) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LayeredImage2D::setPoints(std::vector<std::vector<Point2D>> points) {
  mPoints = points;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LayeredImage2D::setBound(std::string boundType, float min, float max) {
  if (boundType == "x") {
    mBoundX = Bound{min, max};

  } else if (boundType == "y") {
    mBoundY = Bound{min, max};
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<Point2D>> LayeredImage2D::getPoints() {
  return mPoints;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<Bound> LayeredImage2D::getBound(std::string boundType) {
  if (boundType == "x") {
    return mBoundX;

  } else if (boundType == "y") {
    return mBoundY;
  }
}

//Volume3D//////////////////////////////////////////////////////////////////////////////////////////

Volume3D::Volume3D(std::vector<Point3D> points, double timeStamp, Bound boundX, Bound boundY, Bound boundZ, 
  Dimension dimension) 
  : mTimeStamp(timeStamp)
  , mPoints(points)
  , mBoundX(boundX)
  , mBoundY(boundY)
  , mBoundZ(boundZ)
  , mDimension(dimension) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Volume3D::setPoints(std::vector<Point3D> points) {
  mPoints = points;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Volume3D::setBound(std::string boundType, float min, float max) {
  if (boundType == "x") {
    mBoundX = Bound{min, max};

  } else if (boundType == "y") {
    mBoundY = Bound{min, max};

  } else if (boundType == "z") {
    mBoundZ = Bound{min, max};
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<Point3D> Volume3D::getPoints() {
  return mPoints;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<Bound> Volume3D::getBound(std::string boundType) {
  if (boundType == "x") {
    return mBoundX;

  } else if (boundType == "y") {
    return mBoundY;

  } else if (boundType == "z") {
    return mBoundZ;
  }
}

} // namespace csp::visualquery