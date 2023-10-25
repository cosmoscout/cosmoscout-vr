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

#include <glm/glm.hpp>

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

Dimensions::Dimensions(uint32_t width, uint32_t length, uint32_t depth)
    : mWidth(width)
    , mLength(length)
    , mDepth(depth) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::uvec3 Dimensions::getDimensions() const {
  return {mWidth, mLength, mDepth};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Dimensions::setDimensions(uint32_t width, uint32_t length, uint32_t depth) {
  mWidth  = width;
  mLength = length;
  mDepth  = depth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t Dimensions::getWidth() const {
  return mWidth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Dimensions::setWidth(uint32_t width) {
  mWidth = width;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t Dimensions::getLength() const {
  return mLength;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Dimensions::setLength(uint32_t length) {
  mLength = length;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t Dimensions::getDepth() const {
  return mDepth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Dimensions::setDepth(uint32_t depth) {
  mDepth = depth;
}

// Image2D /////////////////////////////////////////////////////////////////////////////////////////

Image2D::Image2D(
    std::vector<Point2D> points, double timeStamp, Bound boundX, Bound boundY, Dimensions dimension)
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
    Bound boundX, Bound boundY, Dimensions dimension)
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
    Bound boundZ, Dimensions dimension)
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