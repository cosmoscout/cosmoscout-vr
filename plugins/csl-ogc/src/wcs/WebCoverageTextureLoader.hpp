////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_TEXTURE_LOADER_HPP
#define CSL_OGC_TEXTURE_LOADER_HPP

#include "csl_ogc_export.hpp"

#include "../common/GDALReader.hpp"
#include "WebCoverage.hpp"
#include "WebCoverageService.hpp"

#include "../../../../src/cs-utils/ThreadPool.hpp"

#include <boost/filesystem.hpp>

#include <array>
#include <map>

namespace csl::ogc {

class GDALReader;

/// Class for requesting map textures from a Web Coverage Services.
class CSL_OGC_EXPORT WebCoverageTextureLoader {
 public:
  /// Struct for defining parameters for a request to a WCS.
  struct Request {
    int32_t  mMaxSize{};
    bool     mKeepAspectRatio{true};
    Bounds2D mBounds;

    std::optional<std::string>         mTime;
    std::optional<std::string>         mFormat;
    std::optional<std::pair<int, int>> mBandRange;

    // This will take precedence over mBandRange
    std::optional<std::vector<int>> mBandList;
  };

  /// Creates a new ThreadPool with the specified amount of threads.
  WebCoverageTextureLoader();

  /// Async WCS texture loader.
  /// Returns an empty optional if loading the texture failed.
  std::future<std::optional<GDALReader::Texture>> loadTextureAsync(WebCoverageService const& wcs,
      WebCoverage const& coverage, Request const& request, std::string const& coverageCache,
      bool saveToCache);

  /// WCS texture loader.
  /// Returns an empty optional if loading the texture failed.
  std::optional<GDALReader::Texture> loadTexture(WebCoverageService const& wcs,
      WebCoverage const& coverage, Request const& request, std::string const& coverageCache,
      bool saveToCache);

 private:
  /// Requests a coverage from a WCS.
  /// Returns a binary stream of the texture file if the request succeeds.
  /// Returns an empty optional if the request fails.
  std::optional<std::stringstream> requestTexture(
      WebCoverageService const& wcs, WebCoverage const& coverage, Request const& request);

  /// Saves a binary stream of a texture file to the given path.
  void saveTextureToFile(boost::filesystem::path const& file, std::stringstream const& data);

  /// Constructs a path for loading/saving the texture requested with the given parameters.
  boost::filesystem::path getCachePath(WebCoverageService const& wcs, WebCoverage const& coverage,
      Request const& request, std::string const& coverageCache);

  /// Constructs a request URL for the given parameters.
  std::string getRequestUrl(
      WebCoverageService const& wcs, WebCoverage const& coverage, Request const& request);

  const std::map<std::string, std::string> mMimeToExtension = {{"image/png", "png"},
      {"image/jpeg", "jpg"}, {"image/jpg", "jpg"}, {"image/tiff", "tiff"},
      {"application/x-netcdf", "nc"}, {"application/x-netcdf4", "nc"}};

  std::mutex            mTextureMutex;
  cs::utils::ThreadPool mThreadPool;
};

} // namespace csl::ogc

#endif // CSL_OGC_TEXTURE_LOADER_HPP
