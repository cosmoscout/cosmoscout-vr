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
  struct GreyScaleTexture {
    int                   x{};
    int                   y{};
    std::array<double, 4> lnglatBounds{};
    std::array<double, 2> dataRange{};
    int                   buffersize{};
    void*                 buffer{};
    int                   timeIndex = 0;
    int                   layers    = 1;
    // The gdal data type of the texture, e.g. Float32, UInt16 etc.
    GDALDataType type{};
    // As buffer is a void pointer, we'll need the size of the underlying type
    float typeSize = 1;
  };

  /**
   * Load all reader DLLs
   */
  static void InitGDAL();

  /**
   * Reads a GDAL supported gray scale image into the texture passed as reference
   */
  static void ReadGrayScaleTexture(GreyScaleTexture& texture, std::string filename, int layer = 1);

  /**
   * Reads a GDAL supported gray scale image from a stream into the texture passed as reference
   */
  static void ReadGrayScaleTexture(GreyScaleTexture& texture, std::stringstream const& data,
      const std::string& filename, int layer = 1);

  /**
   * Get the number of layers in the texture
   */
  static int ReadNumberOfLayers(std::string filename);

  /**
   * Adds a texture with unique path to the cache
   */
  static void AddTextureToCache(const std::string& path, GreyScaleTexture& texture);

  /**
   * Clear cache
   */
  static void ClearCache();

 private:
  /**
   * Warps the given dataset to WGS84, writes the data to "texture" and caches it
   */
  static void BuildTexture(GDALDataset* poDatasetSrc, GreyScaleTexture& texture,
      std::string const& filename, int layer = 1);

  /**
   * Mapping of (virtual) filesystem path to calculated greyscale texture
   */
  static std::map<std::string, GreyScaleTexture> mTextureCache;

  /**
   * Mapping of (virtual) filesystem path to number of bands in a texture
   */
  static std::map<std::string, int> mBandsCache;
  static std::mutex                 mMutex;
  static bool                       mIsInitialized;
};

} // namespace csl::ogc

#endif // CSL_OGC_GDAL_READER
