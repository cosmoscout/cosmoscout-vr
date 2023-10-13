////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "types.hpp"

#include <ctime>
#include <optional>
#include <utility>
#include <vector>

namespace csp::visualquery {

TimeStamp::TimeStamp(double timeStamp)
    : mTimeStamp(timeStamp) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double TimeStamp::getTimeStamp() const {
  return mTimeStamp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeStamp::setTimeStamp(double timeStamp) {
  mTimeStamp = timeStamp;
}

// Dimension/////////////////////////////////////////////////////////////////////////////////////////

Dimension::Dimension(int width, int length, int depth)
    : mWidth(width)
    , mLength(length)
    , mDepth(depth) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<int> Dimension::getDimension(const std::string& dimensionType) {
  if (dimensionType == "width") {
    return std::make_optional(mWidth);
  }
  if (dimensionType == "length") {
    return std::make_optional(mLength);
  }
  if (dimensionType == "depth") {
    return std::make_optional(mDepth);
  }
  return std::nullopt;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Dimension::setDimension(int width, int length, int depth) {
  mWidth  = width;
  mLength = length;
  mDepth  = depth;
}

void Dimension::setDimension(const std::string& dimensionType, int value) {
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

// Image2D///////////////////////////////////////////////////////////////////////////////////////////

Image2D::Image2D(
    std::vector<Point2D> points, double timeStamp, Bound boundX, Bound boundY, Dimension dimension)
    : mTimeStamp(timeStamp)
    , mPoints(std::move(points))
    , mBoundX(boundX)
    , mBoundY(boundY)
    , mDimension(dimension) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Image2D::Image2D()
    : mTimeStamp(0)
    , mBoundX()
    , mBoundY()
    , mDimension(0, 0, 0) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Image2D::setPoints(std::vector<Point2D> points) {
  mPoints = std::move(points);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Image2D::setBound(const std::string& boundType, float min, float max) {
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

std::optional<Bound> Image2D::getBound(const std::string& boundType) {
  if (boundType == "x") {
    return std::make_optional(mBoundX);
  }

  if (boundType == "y") {
    return std::make_optional(mBoundY);
  }
  return std::nullopt;
}

// LayeredImage2D////////////////////////////////////////////////////////////////////////////////////

LayeredImage2D::LayeredImage2D(std::vector<std::vector<Point2D>> points, double timeStamp,
    Bound boundX, Bound boundY, Dimension dimension)
    : mTimeStamp(timeStamp)
    , mPoints(std::move(points))
    , mBoundX(boundX)
    , mBoundY(boundY)
    , mDimension(dimension) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LayeredImage2D::setPoints(std::vector<std::vector<Point2D>> points) {
  mPoints = std::move(points);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LayeredImage2D::setBound(const std::string& boundType, float min, float max) {
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

std::optional<Bound> LayeredImage2D::getBound(const std::string& boundType) {
  if (boundType == "x") {
    return std::make_optional(mBoundX);
  }

  if (boundType == "y") {
    return std::make_optional(mBoundY);
  }
  return std::nullopt;
}

// Volume3D//////////////////////////////////////////////////////////////////////////////////////////

Volume3D::Volume3D(std::vector<Point3D> points, double timeStamp, Bound boundX, Bound boundY,
    Bound boundZ, Dimension dimension)
    : mTimeStamp(timeStamp)
    , mPoints(std::move(points))
    , mBoundX(boundX)
    , mBoundY(boundY)
    , mBoundZ(boundZ)
    , mDimension(dimension) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Volume3D::setPoints(std::vector<Point3D> points) {
  mPoints = std::move(points);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Volume3D::setBound(std::string const& boundType, float min, float max) {
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

std::optional<Bound> Volume3D::getBound(std::string const& boundType) {
  if (boundType == "x") {
    return std::make_optional(mBoundX);
  }

  if (boundType == "y") {
    return std::make_optional(mBoundY);
  }

  if (boundType == "z") {
    return std::make_optional(mBoundZ);
  }
  return std::nullopt;
}

} // namespace csp::visualquery