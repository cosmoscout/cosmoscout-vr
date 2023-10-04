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

TimeStamp::TimeStamp(std::time_t timeStamp) {
  mTimeStamp = timeStamp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::time_t TimeStamp::getTimeStamp() {
    return mTimeStamp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeStamp::setTimeStamp(std::time_t timeStamp) {
    mTimeStamp = timeStamp;
}

//Image2D///////////////////////////////////////////////////////////////////////////////////////////

Image2D::Image2D(std::vector<Point2D> points, std::time_t timeStamp, Bound boundX, Bound boundY) 
 : TimeStamp(timeStamp) {
    mPoints = points;
    mBoundX = boundX;
    mBoundY = boundY;
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

LayeredImage2D::LayeredImage2D(std::vector<std::vector<Point2D>> points, std::time_t timeStamp, Bound boundX, Bound boundY) 
 : TimeStamp(timeStamp) {
    mPoints = points;
    mBoundX = boundX;
    mBoundY = boundY;
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

Volume3D::Volume3D(std::vector<Point3D> points, std::time_t timeStamp,
  Bound boundX, Bound boundY, Bound boundZ) : TimeStamp(timeStamp) {
    mPoints = points;
    mBoundX = boundX;
    mBoundY = boundY;
    mBoundZ = boundZ;
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