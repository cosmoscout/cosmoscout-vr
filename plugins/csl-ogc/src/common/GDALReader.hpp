////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_GDAL_READER
#define CSL_OGC_GDAL_READER

#include "csl_ogc_export.hpp"

#include <array>
#include <map>
#include <mutex>
#include <string>

// GDAL c++ includes
#include <gdal_priv.h>

#include "../logger.hpp"

namespace csl::ogc {

class CSL_OGC_EXPORT GDALReader {
 public:
  /**
   * Struct to store all required information for a float texture
   * e.g. sizes, data ranges, the buffer itself, and geo-referenced bounds
   */
  struct Texture {

    // The width and height of the texture. The width corresponds to the longitude range
    // and the height to the latitude range.
    uint32_t mWidth{};
    uint32_t mHeight{};
    uint32_t mBands{};

    // The geo-referenced bounds of the texture [minX, maxX, minY, maxY].
    std::array<double, 4> mLnglatBounds{};

    // The gdal data type of the texture, e.g. Float32, UInt16 etc.
    GDALDataType mDataType{};

    // The entire data range of all layers.
    std::array<double, 2> mDataRange{};

    // The data ranges of the individual bands.
    std::vector<std::array<double, 2>> mBandDataRanges{};

    // The buffer containing the texture data. If mBands > 1, the data for each band is
    // stored sequentially.
    void*  mBuffer     = nullptr;
    size_t mBufferSize = 0;
  };

  /**
   * Load all reader DLLs
   */
  static void InitGDAL();

  /**
   * Reads a GDAL supported image into the texture passed as reference.
   */
  static void ReadTexture(Texture& texture, std::string filename);

  /**
   * Reads a GDAL supported gray scale image from a stream into the texture passed as reference.
   */
  static void ReadTexture(
      Texture& texture, std::stringstream const& data, const std::string& filename);

  /**
   * Adds a texture with unique key to the cache
   */
  static void AddTextureToCache(std::string const& key, Texture const& texture);

  /**
   * Clear cache
   */
  static void ClearCache();

 private:
  /**
   * Warps the given dataset to WGS84, writes the data to "texture" and caches it.
   */
  static void BuildTexture(GDALDataset* dataset, Texture& texture, std::string const& filename);

  /**
   * Mapping of (virtual) filesystem path to calculated greyscale texture
   */
  static std::map<std::string, Texture> mTextureCache;

  /**
   * Mapping of (virtual) filesystem path to number of bands in a texture
   */
  static std::mutex mMutex;
  static bool       mIsInitialized;
};

} // namespace csl::ogc

#endif // CSL_OGC_GDAL_READER
