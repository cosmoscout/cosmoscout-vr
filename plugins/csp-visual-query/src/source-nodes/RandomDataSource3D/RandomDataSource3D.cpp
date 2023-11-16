////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "RandomDataSource3D.hpp"

#include "../../logger.hpp"
#include "../../../../src/cs-utils/filesystem.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string RandomDataSource3D::sName = "RandomDataSource3D";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string RandomDataSource3D::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/RandomDataSource3D.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<RandomDataSource3D> RandomDataSource3D::sCreate() {
  return std::make_unique<RandomDataSource3D>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RandomDataSource3D::RandomDataSource3D() noexcept {
  std::random_device randomDevice;
  mRandomNumberGenerator = std::mt19937(randomDevice());
  mDistribution          = std::uniform_real_distribution();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RandomDataSource3D::~RandomDataSource3D() noexcept = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& RandomDataSource3D::getName() const noexcept {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RandomDataSource3D::process() noexcept {
  if (mBounds.mMaxLat < mBounds.mMinLat || mBounds.mMaxLon < mBounds.mMinLon) {
    return;
  }

  F32ValueVector points;
  points.reserve(static_cast<size_t>(
      (mBounds.mMaxLat - mBounds.mMinLat) * (mBounds.mMaxLon - mBounds.mMinLon)));

  for (double lat = mBounds.mMinLat; lat <= mBounds.mMaxLat; lat += 1.0) {
    for (double lon = mBounds.mMinLon; lon <= mBounds.mMaxLon; lon += 1.0) {
      for (double height = mBounds.mMinHeight; height <= mBounds.mMaxHeight; height += 1000.0) {
        points.emplace_back(std::vector{static_cast<float>(mDistribution(mRandomNumberGenerator)),
            static_cast<float>(mDistribution(mRandomNumberGenerator)),
            static_cast<float>(mDistribution(mRandomNumberGenerator)),
            static_cast<float>(mDistribution(mRandomNumberGenerator))});
      }
    }
  }

  mData = std::make_shared<Volume3D>(points, 4,
      glm::uvec3{static_cast<uint32_t>(mBounds.mMaxLon - mBounds.mMinLon),
          static_cast<uint32_t>(mBounds.mMaxLat - mBounds.mMinLat),
          static_cast<uint32_t>((mBounds.mMaxHeight - mBounds.mMinHeight) / 1000.0)},
      mBounds, std::nullopt);

  writeOutput("Volume3D", mData);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RandomDataSource3D::onMessageFromJS(nlohmann::json const& message) {
  mBounds = message.get<csl::ogc::Bounds3D>();

  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json RandomDataSource3D::getData() const {
  return mBounds;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RandomDataSource3D::setData(nlohmann::json const& json) {
  mBounds = json.get<csl::ogc::Bounds3D>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery