////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "RandomDataSource2D.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string RandomDataSource2D::sName = "RandomDataSource2D";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string RandomDataSource2D::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/RandomDataSource2D.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<RandomDataSource2D> RandomDataSource2D::sCreate() {
  return std::make_unique<RandomDataSource2D>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RandomDataSource2D::RandomDataSource2D() noexcept {
  std::random_device randomDevice;
  mRandomNumberGenerator = std::mt19937(randomDevice());
  mDistribution          = std::uniform_real_distribution();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RandomDataSource2D::~RandomDataSource2D() noexcept = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& RandomDataSource2D::getName() const noexcept {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RandomDataSource2D::process() noexcept {
  if (mBounds.mMaxLat < mBounds.mMinLat || mBounds.mMaxLon < mBounds.mMinLon) {
    return;
  }

  F32ValueVector points;
  points.reserve(static_cast<size_t>(
      (mBounds.mMaxLat - mBounds.mMinLat) * (mBounds.mMaxLon - mBounds.mMinLon)));

  for (double lat = mBounds.mMinLat; lat <= mBounds.mMaxLat; lat += 1.0) {
    for (double lon = mBounds.mMinLon; lon <= mBounds.mMaxLon; lon += 1.0) {
      points.emplace_back(std::vector{static_cast<float>(mDistribution(mRandomNumberGenerator)),
          static_cast<float>(mDistribution(mRandomNumberGenerator)),
          static_cast<float>(mDistribution(mRandomNumberGenerator)),
          static_cast<float>(mDistribution(mRandomNumberGenerator))});
    }
  }

  mData = std::make_shared<Image2D>(points, 4,
      glm::uvec2{static_cast<uint32_t>((mBounds.mMaxLon - mBounds.mMinLon)),
          static_cast<uint32_t>((mBounds.mMaxLat - mBounds.mMinLat))},
      mBounds, glm::vec2(0, 1), std::nullopt);

  writeOutput("Image2D", mData);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RandomDataSource2D::onMessageFromJS(nlohmann::json const& message) {
  mBounds = message.get<csl::ogc::Bounds2D>();

  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json RandomDataSource2D::getData() const {
  return mBounds;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RandomDataSource2D::setData(nlohmann::json const& json) {
  mBounds = json.get<csl::ogc::Bounds2D>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery