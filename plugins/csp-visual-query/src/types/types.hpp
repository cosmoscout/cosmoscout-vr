////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_TYPES_HPP
#define CSP_VISUAL_QUERY_TYPES_HPP

#include <ctime>
#include <string>
#include <vector>

namespace csp::visualquery {

struct Point2D {
  float              x; // longitude
  float              y; // latitude
  std::vector<float> value;
};

struct Point3D {
  float              x; // longitude
  float              y; // latitude
  float              z; // height
  std::vector<float> value;
};

struct Bound {
  float min;
  float max;
};

struct Dimensions {
 public:
  Dimensions(uint32_t width, uint32_t length, uint32_t depth = 1);

  glm::uvec3 getDimensions() const;
  void       setDimensions(uint32_t width, uint32_t length, uint32_t depth = 1);

  uint32_t getWidth() const;
  void     setWidth(uint32_t width);

  uint32_t getLength() const;
  void     setLength(uint32_t length);

  uint32_t getDepth() const;
  void     setDepth(uint32_t depth);

 private:
  uint32_t mWidth;
  uint32_t mLength;
  uint32_t mDepth;
};

struct TimeStamp {
 public:
  TimeStamp(double timeStamp);

  double getTimeStamp() const;
  void   setTimeStamp(double timeStamp);

 private:
  double mTimeStamp; // time in TDB
};

class Image2D {
 public:
  Image2D(std::vector<Point2D> points, double timeStamp, Bound boundX, Bound boundY,
      Dimensions dimension);
  Image2D();

  std::vector<Point2D> getPoints();
  std::optional<Bound> getBound(const std::string& boundType);

  void setPoints(std::vector<Point2D> points);
  void setBound(const std::string& boundType, float min, float max);

  std::optional<double> mTimeStamp;
  Dimensions            mDimension;

 private:
  std::vector<Point2D> mPoints;
  Bound                mBoundX;
  Bound                mBoundY;
};

class LayeredImage2D {
 public:
  LayeredImage2D(std::vector<std::vector<Point2D>> points, double timeStamp, Bound boundX,
      Bound boundY, Dimensions dimension);

  std::vector<std::vector<Point2D>> getPoints();
  std::optional<Bound>              getBound(const std::string& boundType);

  void setPoints(std::vector<std::vector<Point2D>> points);
  void setBound(const std::string& boundType, float min, float max);

  TimeStamp mTimeStamp;
  Dimensions mDimension;

 private:
  std::vector<std::vector<Point2D>> mPoints;
  Bound                             mBoundX;
  Bound                             mBoundY;
};

class Volume3D {
 public:
  Volume3D(std::vector<Point3D> points, double timeStamp, Bound boundX, Bound boundY, Bound boundZ,
      Dimensions dimension);

  std::vector<Point3D> getPoints();
  std::optional<Bound> getBound(const std::string& boundType);

  void setPoints(std::vector<Point3D> points);
  void setBound(const std::string& boundType, float min, float max);

  TimeStamp mTimeStamp;
  Dimensions mDimension;

 private:
  std::vector<Point3D> mPoints;
  Bound                mBoundX;
  Bound                mBoundY;
  Bound                mBoundZ;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_TYPES_HPP