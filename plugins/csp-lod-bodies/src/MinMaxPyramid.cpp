////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "MinMaxPyramid.hpp"
#include "Tile.hpp"

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

MinMaxPyramid::MinMaxPyramid(Tile<float>* tile)
    : mTileResolution(tile->getResolution())
    , mLevels(std::log2(mTileResolution) - 1)
    , mMinPyramid(mLevels)
    , mMaxPyramid(mLevels) {

  for (uint32_t i(0); i < mLevels; ++i) {
    uint32_t resolution = mTileResolution / (1 << (i + 1));
    mMinPyramid[i] = std::vector<float>(resolution * resolution, std::numeric_limits<float>::max());
    mMaxPyramid[i] =
        std::vector<float>(resolution * resolution, std::numeric_limits<float>::lowest());
  }

  // mMinPyramid[0] = std::vector<float>(128 * 128, std::numeric_limits<float>::max());
  // mMinPyramid[1] = std::vector<float>(64 * 64, std::numeric_limits<float>::max());
  // mMinPyramid[2] = std::vector<float>(32 * 32, std::numeric_limits<float>::max());
  // mMinPyramid[3] = std::vector<float>(16 * 16, std::numeric_limits<float>::max());
  // mMinPyramid[4] = std::vector<float>(8 * 8, std::numeric_limits<float>::max());
  // mMinPyramid[5] = std::vector<float>(4 * 4, std::numeric_limits<float>::max());
  // mMinPyramid[6] = std::vector<float>(2 * 2, std::numeric_limits<float>::max());

  // mMaxPyramid[0] = std::vector<float>(128 * 128, -std::numeric_limits<float>::max());
  // mMaxPyramid[1] = std::vector<float>(64 * 64, -std::numeric_limits<float>::max());
  // mMaxPyramid[2] = std::vector<float>(32 * 32, -std::numeric_limits<float>::max());
  // mMaxPyramid[3] = std::vector<float>(16 * 16, -std::numeric_limits<float>::max());
  // mMaxPyramid[4] = std::vector<float>(8 * 8, -std::numeric_limits<float>::max());
  // mMaxPyramid[5] = std::vector<float>(4 * 4, -std::numeric_limits<float>::max());
  // mMaxPyramid[6] = std::vector<float>(2 * 2, -std::numeric_limits<float>::max());

  // Construct first MinMaxPyramid layer by sampling 2x2 values
  uint32_t halfSize = static_cast<uint32_t>((mTileResolution - 1) / 2);
  uint32_t x2       = 0;
  uint32_t y2       = 0;

  for (uint32_t y = 0; y < mTileResolution; ++y) {
    x2 = 0;
    for (uint32_t x = 0; x < mTileResolution; ++x) {
      float const v = tile->data()[y * mTileResolution + x];

      mMinValue = std::min(mMinValue, v);
      mMaxValue = std::max(mMaxValue, v);
      mAvgValue += v / (mTileResolution * mTileResolution);

      x2                                 = std::min(x2, halfSize - 1);
      y2                                 = std::min(y2, halfSize - 1);
      mMinPyramid[0][y2 * halfSize + x2] = std::min(mMinPyramid[0][y2 * halfSize + x2], v);
      mMaxPyramid[0][y2 * halfSize + x2] = std::max(mMaxPyramid[0][y2 * halfSize + x2], v);

      if (x % 2 == 1) {
        x2 += 1;
      }
    }

    if (y % 2 == 1) {
      y2 += 1;
    }
  }

  // Build remaining MinMaxPyramid layers
  for (uint32_t i(1); i < mLevels; ++i) {
    y2 = 0;
    for (uint32_t y = 0; y < (mTileResolution - 1) * std::pow(0.5, i); ++y) {
      x2 = 0;
      for (uint32_t x = 0; x < (mTileResolution - 1) * std::pow(0.5, i); ++x) {
        mMinPyramid[i][static_cast<uint64_t>(y2 * halfSize * std::pow(0.5, i) + x2)] = std::min(
            mMinPyramid[i - 1]
                       [static_cast<uint64_t>(y * (mTileResolution - 1) * std::pow(0.5, i) + x)],
            mMinPyramid[i][static_cast<uint64_t>(y2 * halfSize * std::pow(0.5, i) + x2)]);
        mMaxPyramid[i][static_cast<uint64_t>(y2 * halfSize * std::pow(0.5, i) + x2)] = std::max(
            mMaxPyramid[i - 1]
                       [static_cast<uint64_t>(y * (mTileResolution - 1) * std::pow(0.5, i) + x)],
            mMaxPyramid[i][static_cast<uint64_t>(y2 * halfSize * std::pow(0.5, i) + x2)]);
        if (x % 2 == 1) {
          x2 += 1;
        }
      }
      if (y % 2 == 1) {
        y2 += 1;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////


std::vector<std::vector<float>>& MinMaxPyramid::getMinPyramid() {
  return mMinPyramid;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<float>>& MinMaxPyramid::getMaxPyramid() {
  return mMaxPyramid;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float MinMaxPyramid::getMin(std::vector<int> const& quadrants) {
  return getData(mMinPyramid, quadrants);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float MinMaxPyramid::getMax(std::vector<int> const& quadrants) {
  return getData(mMaxPyramid, quadrants);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float MinMaxPyramid::getData(
    std::vector<std::vector<float>>& pyramid, std::vector<int> const& quadrants) {
  // number of pyramid levels:
  auto qSize = quadrants.size();
  // particular pyramid layer,
  // where min max information of the corresponding tile resolution is stored
  auto layerId = 7 - qSize;
  // pyramid layer address of searched value
  int x(0);
  int y(0);
  for (size_t i(0); i < qSize; ++i) {
    int step = int(std::pow(2, qSize - (i + 1)));
    switch (quadrants[i]) {
    case 1:
      x += step;
      break;
    case 2:
      y += step;
      break;
    case 3:
      x += step;
      y += step;
      break;
    }
  }

  return pyramid[layerId][int(y * std::pow(2, qSize) + x)];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
