////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_TYPES_HPP
#define CSP_VISUAL_QUERY_TYPES_HPP

#include <ctime>
#include <stdint.h>
#include <string>
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

struct Image2D {
  Image2D() = default;

  Image2D(PointsType points, glm::uvec2 dimension, csl::ogc::Bounds bounds,
      std::optional<csl::ogc::TimeInterval> timeStamp = std::nullopt)
      : mPoints(std::move(points))
      , mDimension(dimension)
      , mBounds(bounds)
      , mTimeStamp(timeStamp) {
  }

  PointsType mPoints;

  glm::uvec2       mDimension;
  csl::ogc::Bounds mBounds;

  std::optional<csl::ogc::TimeInterval> mTimeStamp;
};

struct Volume3D {
  PointsType mPoints;

  glm::uvec3       mDimension;
  csl::ogc::Bounds mBounds;

  std::optional<csl::ogc::TimeInterval> mTimeStamp;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_TYPES_HPP