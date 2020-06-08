////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MinMaxPyramid.hpp"
#include "Tile.hpp"

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

MinMaxPyramid::MinMaxPyramid()
    : mMinPyramid(7, std::vector<float>())
    , mMaxPyramid(7, std::vector<float>()) {
  mMinPyramid[0] = std::vector<float>(128 * 128, std::numeric_limits<float>::max());
  mMinPyramid[1] = std::vector<float>(64 * 64, std::numeric_limits<float>::max());
  mMinPyramid[2] = std::vector<float>(32 * 32, std::numeric_limits<float>::max());
  mMinPyramid[3] = std::vector<float>(16 * 16, std::numeric_limits<float>::max());
  mMinPyramid[4] = std::vector<float>(8 * 8, std::numeric_limits<float>::max());
  mMinPyramid[5] = std::vector<float>(4 * 4, std::numeric_limits<float>::max());
  mMinPyramid[6] = std::vector<float>(2 * 2, std::numeric_limits<float>::max());

  mMaxPyramid[0] = std::vector<float>(128 * 128, -std::numeric_limits<float>::max());
  mMaxPyramid[1] = std::vector<float>(64 * 64, -std::numeric_limits<float>::max());
  mMaxPyramid[2] = std::vector<float>(32 * 32, -std::numeric_limits<float>::max());
  mMaxPyramid[3] = std::vector<float>(16 * 16, -std::numeric_limits<float>::max());
  mMaxPyramid[4] = std::vector<float>(8 * 8, -std::numeric_limits<float>::max());
  mMaxPyramid[5] = std::vector<float>(4 * 4, -std::numeric_limits<float>::max());
  mMaxPyramid[6] = std::vector<float>(2 * 2, -std::numeric_limits<float>::max());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

MinMaxPyramid::MinMaxPyramid(Tile<float>* tile)
    : mMinPyramid(7, std::vector<float>())
    , mMaxPyramid(7, std::vector<float>()) {
  mMinPyramid[0] = std::vector<float>(128 * 128, std::numeric_limits<float>::max());
  mMinPyramid[1] = std::vector<float>(64 * 64, std::numeric_limits<float>::max());
  mMinPyramid[2] = std::vector<float>(32 * 32, std::numeric_limits<float>::max());
  mMinPyramid[3] = std::vector<float>(16 * 16, std::numeric_limits<float>::max());
  mMinPyramid[4] = std::vector<float>(8 * 8, std::numeric_limits<float>::max());
  mMinPyramid[5] = std::vector<float>(4 * 4, std::numeric_limits<float>::max());
  mMinPyramid[6] = std::vector<float>(2 * 2, std::numeric_limits<float>::max());

  mMaxPyramid[0] = std::vector<float>(128 * 128, -std::numeric_limits<float>::max());
  mMaxPyramid[1] = std::vector<float>(64 * 64, -std::numeric_limits<float>::max());
  mMaxPyramid[2] = std::vector<float>(32 * 32, -std::numeric_limits<float>::max());
  mMaxPyramid[3] = std::vector<float>(16 * 16, -std::numeric_limits<float>::max());
  mMaxPyramid[4] = std::vector<float>(8 * 8, -std::numeric_limits<float>::max());
  mMaxPyramid[5] = std::vector<float>(4 * 4, -std::numeric_limits<float>::max());
  mMaxPyramid[6] = std::vector<float>(2 * 2, -std::numeric_limits<float>::max());

  int HalfSizeX = static_cast<int32_t>((TileBase::SizeX - 1) * 0.5); // 128
  int x2        = 0;                                                 // 0..128
  int y2        = 0;                                                 // 0..128

  for (int y = 0; y < TileBase::SizeY; ++y) {
    x2 = 0;
    for (int x = 0; x < TileBase::SizeX; ++x) {
      float const v = tile->data()[y * TileBase::SizeX + x];

      mMinValue = std::min(mMinValue, v);
      mMaxValue = std::max(mMaxValue, v);
      mAvgValue += v / (TileBase::SizeY * TileBase::SizeY);

      // Construct first 128x128 MinMaxPyramid layer by sampling 256x256 values
      x2                                  = std::min(x2, 127);
      y2                                  = std::min(y2, 127);
      mMinPyramid[0][y2 * HalfSizeX + x2] = std::min(mMinPyramid[0][y2 * HalfSizeX + x2], v);
      mMaxPyramid[0][y2 * HalfSizeX + x2] = std::max(mMaxPyramid[0][y2 * HalfSizeX + x2], v);

      // 256 -> 128
      if (x % 2 == 1) {
        x2 += 1;
      }
    }

    // 256 -> 128
    if (y % 2 == 1) {
      y2 += 1;
    }
  }

  // Build remaining MinMaxPyramid layers 64x62-2x2
  for (int i(1); i < 7; ++i) {
    y2 = 0;
    for (int y = 0; y < (TileBase::SizeY - 1) * std::pow(0.5, i); ++y) {
      x2 = 0;
      for (int x = 0; x < (TileBase::SizeX - 1) * std::pow(0.5, i); ++x) {
        mMinPyramid[i][static_cast<uint64_t>(y2 * HalfSizeX * std::pow(0.5, i) + x2)] = std::min(
            mMinPyramid[i - 1]
                       [static_cast<uint64_t>(y * (TileBase::SizeX - 1) * std::pow(0.5, i) + x)],
            mMinPyramid[i][static_cast<uint64_t>(y2 * HalfSizeX * std::pow(0.5, i) + x2)]);
        mMaxPyramid[i][static_cast<uint64_t>(y2 * HalfSizeX * std::pow(0.5, i) + x2)] = std::max(
            mMaxPyramid[i - 1]
                       [static_cast<uint64_t>(y * (TileBase::SizeX - 1) * std::pow(0.5, i) + x)],
            mMaxPyramid[i][static_cast<uint64_t>(y2 * HalfSizeX * std::pow(0.5, i) + x2)]);
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

MinMaxPyramid::~MinMaxPyramid() {
  for (int layer(0); layer < 7; ++layer) {
    mMinPyramid[layer].clear();
    mMaxPyramid[layer].clear();
  }

  mMinPyramid.clear();
  mMaxPyramid.clear();
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
