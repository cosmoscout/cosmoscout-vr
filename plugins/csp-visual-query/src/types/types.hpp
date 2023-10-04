////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_TYPES_HPP
#define CSP_VISUAL_QUERY_TYPES_HPP

#include <string>
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
  TimeStamp(std::time_t timeStamp);
  std::time_t getTimeStamp();
  void setTimeStamp(std::time_t timeStamp); // ???

 private:
  std::time_t mTimeStamp;
};

class Image2D : public TimeStamp {
 public:
  Image2D(std::vector<Point2D> points, std::time_t timeStamp, Bound boundX, Bound boundY);

  std::vector<Point2D> getPoints();
  std::optional<Bound> getBound(std::string boundType);

  void setPoints(std::vector<Point2D> points);
  void setBound(std::string boundType, float min, float max);

 private:
  std::vector<Point2D> mPoints;
  Bound mBoundX;
  Bound mBoundY;
};

class LayeredImage2D : public TimeStamp {
 public:
  LayeredImage2D(std::vector<std::vector<Point2D>> points, std::time_t timeStamp, Bound boundX, Bound boundY);

  std::vector<std::vector<Point2D>> getPoints();
  std::optional<Bound>              getBound(std::string boundType);

  void setPoints(std::vector<std::vector<Point2D>> points);
  void setBound(std::string boundType, float min, float max);

 private:
  std::vector<std::vector<Point2D>> mPoints;
  Bound mBoundX;
  Bound mBoundY;
};

class Volume3D : public TimeStamp {
 public:
  Volume3D(std::vector<Point3D> points, std::time_t timeStamp);
  Volume3D(std::vector<Point3D> points, std::time_t timeStamp, Bound boundX, Bound boundY, Bound boundZ);

  std::vector<Point3D> getPoints();
  std::optional<Bound> getBound(std::string boundType);

  void setPoints(std::vector<Point3D> points);
  void setBound(std::string boundType, float min, float max);

 private:
  std::vector<Point3D> mPoints;
  Bound mBoundX;
  Bound mBoundY;
  Bound mBoundZ;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_TYPES_HPP