////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_TYPES_HPP
#define CSP_VISUAL_QUERY_TYPES_HPP

#include "../../../csl-ogc/src/common/utils.hpp"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace csp::visualquery {

template <typename T>
using PointValues = std::vector<T>;

using U8ValueVector  = std::vector<PointValues<uint8_t>>;
using U16ValueVector = std::vector<PointValues<uint16_t>>;
using U32ValueVector = std::vector<PointValues<uint32_t>>;

using I16ValueVector = std::vector<PointValues<int16_t>>;
using I32ValueVector = std::vector<PointValues<int32_t>>;

using F32ValueVector = std::vector<PointValues<float>>;

using PointsType = std::variant<U8ValueVector, U16ValueVector, U32ValueVector, I16ValueVector,
    I32ValueVector, F32ValueVector>;

struct Image1D {
  Image1D() = default;

  Image1D(PointsType points, size_t numScalars, uint32_t dimension, glm::dvec2 longLat,
      std::optional<csl::ogc::TimeInterval> timeStamp = std::nullopt)
      : mPoints(std::move(points))
      , mNumScalars(numScalars)
      , mDimension(dimension)
      , mLongLat(longLat)
      , mTimeStamp(std::move(timeStamp)) {
  }

  PointsType mPoints;
  size_t     mNumScalars{};

  uint32_t   mDimension{};
  glm::dvec2 mLongLat;

  std::optional<csl::ogc::TimeInterval> mTimeStamp;

  template <typename T>
  T at(uint32_t x) {
    return std::get<T>(mPoints).at(x);
  }
};

struct Image2D {
  Image2D() = default;

  Image2D(PointsType points, size_t numScalars, glm::uvec2 dimension, csl::ogc::Bounds2D bounds,
      glm::vec2 minMax, std::optional<csl::ogc::TimeInterval> timeStamp = std::nullopt)
      : mPoints(std::move(points))
      , mNumScalars(numScalars)
      , mDimension(dimension)
      , mBounds(bounds)
      , mTimeStamp(std::move(timeStamp))
      , mMinMax(minMax) {
  }

  PointsType mPoints;
  size_t     mNumScalars{};

  glm::uvec2         mDimension{};
  csl::ogc::Bounds2D mBounds;

  std::optional<csl::ogc::TimeInterval> mTimeStamp;
  glm::dvec2                            mMinMax;

  template <typename T>
  T at(uint32_t x, uint32_t y) {
    return std::get<T>(mPoints).at(y * mDimension.x + x);
  }
};

struct Volume3D {
  Volume3D() = default;

  Volume3D(PointsType points, size_t numScalars, glm::uvec3 dimension, csl::ogc::Bounds3D bounds,
      std::optional<csl::ogc::TimeInterval> timeStamp = std::nullopt)
      : mPoints(std::move(points))
      , mNumScalars(numScalars)
      , mDimension(dimension)
      , mBounds(bounds)
      , mTimeStamp(std::move(timeStamp)) {
  }

  PointsType mPoints;
  size_t     mNumScalars{};

  glm::uvec3         mDimension{};
  csl::ogc::Bounds3D mBounds;

  std::optional<csl::ogc::TimeInterval> mTimeStamp;

  template <typename T>
  auto at(uint32_t x, uint32_t y, uint32_t z) {
    return std::get<T>(mPoints).at(x + (y * mDimension.x) + (z * mDimension.x * mDimension.y));
  }
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_TYPES_HPP