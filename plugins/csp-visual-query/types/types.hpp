////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_IMAGE_HPP
#define CSP_VISUAL_QUERY_IMAGE_HPP

#include <vector>
#include <ctime>

namespace csp::visualquery {

struct Point2D {
  float x; // longitude
  float y; // latitude
  std::vector<float> value;
};

struct Point3D {
  float x; // longitude
  float y; // latitude
  float z; // height
  std::vector<float> value;
};

struct Bound {
  float min;
  float max;
};

struct TimeStamp {
 public:
  TimeStamp(double timeStamp);

  double getTimeStamp();
  void setTimeStamp(double timeStamp);

 private:
  double mTimeStamp; // time in TDB
};

class Image2D {
 public:
  Image2D(std::vector<Point2D> points, double timeStamp, Bound boundX, Bound boundY, Dimension dimension);

  std::vector<Point2D> getPoints();
  std::optional<Bound> getBound(std::string boundType);

  void setPoints(std::vector<Point2D> points);
  void setBound(std::string boundType, float min, float max);

  TimeStamp            mTimeStamp;
 private:
  std::vector<Point2D> mPoints;
  Bound mBoundX;
  Bound mBoundY;
};

class LayeredImage2D {
 public:
  LayeredImage2D(std::vector<std::vector<Point2D>> points, double timeStamp, Bound boundX, Bound boundY,

  std::vector<std::vector<Point2D>> getPoints();
  std::optional<Bound>              getBound(std::string boundType);

  void setPoints(std::vector<std::vector<Point2D>> points);
  void setBound(std::string boundType, float min, float max);

  TimeStamp                         mTimeStamp;
 private:
  std::vector<std::vector<Point2D>> mPoints;
  Bound mBoundX;
  Bound mBoundY;
};

class Volume3D {
 public:
  Volume3D(std::vector<Point3D> points, double timeStamp, Bound boundX, Bound boundY, Bound boundZ,
  Volume3D(std::vector<Point3D> points, std::time_t timeStamp, Bound boundX, Bound boundY, Bound boundZ);

  std::vector<Point3D> getPoints();
  std::optional<Bound> getBound(std::string boundType);

  void setPoints(std::vector<Point3D> points);
  void setBound(std::string boundType, float min, float max);

  TimeStamp            mTimeStamp;
 private:
  std::vector<Point3D> mPoints;
  Bound mBoundX;
  Bound mBoundY;
  Bound mBoundZ;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_IMAGE_HPP