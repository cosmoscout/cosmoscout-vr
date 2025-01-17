////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WebCoverageTextureLoader.hpp"

#include "WebCoverageException.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm/replace_copy_if.hpp>
#include <curlpp/Infos.hpp>
#include <curlpp/Options.hpp>
#include <fstream>

namespace csl::ogc {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverageTextureLoader::WebCoverageTextureLoader()
    : mThreadPool(std::thread::hardware_concurrency()) {
  GDALReader::InitGDAL();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::optional<GDALReader::Texture>> WebCoverageTextureLoader::loadTextureAsync(
    WebCoverageService const& wcs, WebCoverage const& coverage, Request const& request,
    std::string const& coverageCache, bool saveToCache) {
  return mThreadPool.enqueue(
      [=]() { return loadTexture(wcs, coverage, request, coverageCache, saveToCache); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<GDALReader::Texture> WebCoverageTextureLoader::loadTexture(
    WebCoverageService const& wcs, WebCoverage const& coverage, Request const& request,
    std::string const& coverageCache, bool saveToCache) {
  boost::filesystem::path cachePath = getCachePath(wcs, coverage, request, coverageCache);

  std::optional<std::stringstream> textureStream;
  GDALReader::Texture              texture;

  if (saveToCache && boost::filesystem::exists(cachePath) &&
      boost::filesystem::file_size(cachePath) > 0) {
    GDALReader::ReadTexture(texture, cachePath.string());
  } else {
    textureStream = requestTexture(wcs, coverage, request);
    if (!textureStream.has_value()) {
      return {};
    }

    GDALReader::ReadTexture(texture, textureStream.value(), cachePath.string());

    if (saveToCache) {
      textureStream.value().rdbuf()->pubseekpos(0);
      saveTextureToFile(cachePath, textureStream.value());
    }
  }

  if (!texture.mBuffer) {
    return {};
  }

  return {texture};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::stringstream> WebCoverageTextureLoader::requestTexture(
    WebCoverageService const& wcs, WebCoverage const& coverage, Request const& request) {

  std::string url = getRequestUrl(wcs, coverage, request);

  logger().debug("Performing WCS request '{}'.", url);

  int maxRetries = 3;
  for (int i = 0; i < maxRetries; i++) {
    if (i > 0) {
      logger().debug("Retrying...");
    }

    std::stringstream out(std::ios_base::out | std::ios_base::in | std::ios_base::binary);

    curlpp::Easy curlRequest;
    curlRequest.setOpt(curlpp::options::Url(url));
    curlRequest.setOpt(curlpp::options::WriteStream(&out));
    curlRequest.setOpt(curlpp::options::NoSignal(true));
    curlRequest.setOpt(curlpp::options::SslVerifyPeer(false));
    curlRequest.setOpt(curlpp::options::FollowLocation(true));

    try {
      curlRequest.perform();
    } catch (std::exception& e) {
      logger().warn("Failed to perform WCS request '{}': '{}'!", url, e.what());
      continue;
    }

    std::string contentType = curlpp::Info<CURLINFO_CONTENT_TYPE, std::string>::get(curlRequest);
    // Remove suffix and parameter from content type
    size_t suffixPos    = contentType.find('+');
    size_t parameterPos = contentType.find(';');
    if (suffixPos != std::string::npos) {
      contentType = contentType.substr(0, suffixPos);
    } else if (parameterPos != std::string::npos) {
      contentType = contentType.substr(0, parameterPos);
    }
    if (contentType == "NULL") {
      // No content type was set in the response. This error typically persists only for a short
      // amount of time, so the request can be retried.
      logger().debug("Could not determine response content type.");
      continue;
    }
    if (contentType == "application/xml") {
      // A WCS exception might have occurred.
      try {
        // If there was a valid WCS exception, the problem probably can't be fixed with a retry.
        // => Return an empty object to cancel the request.
        WebCoverageExceptionReport e(out.str());
        logger().warn("WCS Exception occurred for WCS request '{}': '{}'!", url, e.what());
        return {};
      } catch (std::exception const& e) {
        // If parsing the document fails, this might be due to connection problems
        // or corrupted data.
        // => Retry the request.
        logger().debug("Could not create WebCoverageExceptionReport: '{}'.", e.what());
        continue;
      }
    } else if (contentType != request.mFormat.value_or("image/tiff")) {
      logger().debug("Received response of invalid MIME type '{}'.", contentType);
      continue;
    }
    return std::move(out);
  }

  logger().warn("Could not get a valid response for WCS request '{}'!", url);
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebCoverageTextureLoader::saveTextureToFile(
    boost::filesystem::path const& file, std::stringstream const& data) {
  {
    std::unique_lock<std::mutex> lock(mTextureMutex);

    if (boost::filesystem::exists(file) && boost::filesystem::file_size(file) == 0) {
      boost::filesystem::remove(file);
    }

    auto cacheDirPath(boost::filesystem::absolute(file.branch_path()));
    if (!(boost::filesystem::exists(file))) {
      try {
        cs::utils::filesystem::createDirectoryRecursively(
            cacheDirPath, boost::filesystem::perms::all_all);
      } catch (std::exception& e) {
        logger().warn("Failed to create cache directory: '{}'!", e.what());
        return;
      }
    }
  }

  {
    std::ofstream out;
    out.open(file.string(), std::ofstream::out | std::ofstream::binary);

    if (!out) {
      logger().warn("Failed to open '{}' for writing!", file.string());
      return;
    }

    out << data.rdbuf();
  }

  boost::filesystem::perms filePerms =
      boost::filesystem::perms::owner_read | boost::filesystem::perms::owner_write |
      boost::filesystem::perms::group_read | boost::filesystem::perms::group_write |
      boost::filesystem::perms::others_read | boost::filesystem::perms::others_write;
  boost::filesystem::permissions(file, filePerms);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

boost::filesystem::path WebCoverageTextureLoader::getCachePath(WebCoverageService const& wcs,
    WebCoverage const& coverage, Request const& request, std::string const& coverageCache) {

  // Replace forbidden characters in coverage string before creating cache dir.
  std::string coverageFixed;
  boost::replace_copy_if(
      coverage.getId(), std::back_inserter(coverageFixed), boost::is_any_of("*.,:[|]\""), '_');

  // Set file format to three characters.
  std::string fileFormat = mMimeToExtension.at(request.mFormat.value_or("image/tiff"));

  std::stringstream cacheDir;
  cacheDir << coverageCache << "/" << coverageFixed << "/";
  cacheDir << request.mMaxSize << "px/";

  if (request.mTime.has_value()) {
    std::string       year;
    std::stringstream time_stringstream(request.mTime.value());

    // Create dir for year.
    std::getline(time_stringstream, year, '-');
    cacheDir << year << "/";
  }

  std::stringstream cacheFile;
  cacheFile << cacheDir.str() << coverageFixed;

  if (request.mTime.has_value()) {
    cacheFile << "_" << request.mTime.value();
  }

  if (request.mLayerRange.has_value()) {
    cacheFile << "_" << std::to_string(request.mLayerRange.value().first) << "_"
              << std::to_string(request.mLayerRange.value().second);
  }

  // Add Bound string to cache file name
  cacheFile << "_" << utils::toStringWithoutTrailing(request.mBounds.mMinLon) << "_"
            << utils::toStringWithoutTrailing(request.mBounds.mMaxLon) << "_"
            << utils::toStringWithoutTrailing(request.mBounds.mMinLat) << "_"
            << utils::toStringWithoutTrailing(request.mBounds.mMaxLat);

  // Add time string to cache file name if time is specified
  if (request.mTime.has_value()) {
    std::string timeForFile = request.mTime.value();
    std::replace(timeForFile.begin(), timeForFile.end(), '/', '-');
    std::replace(timeForFile.begin(), timeForFile.end(), ':', '-');

    cacheFile << timeForFile;
  }

  cacheFile << "." << fileFormat;

  return boost::filesystem::path(cacheFile.str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebCoverageTextureLoader::getRequestUrl(
    WebCoverageService const& wcs, WebCoverage const& coverage, Request const& request) {

  std::stringstream url;
  url.precision(std::numeric_limits<double>::max_digits10);

  url << wcs.getUrl();
  url << "&SERVICE=WCS";
  url << "&VERSION=2.0.1";
  url << "&REQUEST=GetCoverage";
  url << "&COVERAGEID=" << coverage.getId();

  /// All special chars need to be in url encoded form
  /// This is only really an issue with tomcat servers

  if (request.mBounds != coverage.getSettings().mBounds && request.mBounds != Bounds2D()) {
    // &SUBSET=y(...,...)
    url << "&SUBSET=Lat%28" << request.mBounds.mMinLat << "," << request.mBounds.mMaxLat << "%29";
    // &SUBSET=x(...,...)
    url << "&SUBSET=Long%28" << request.mBounds.mMinLon << "," << request.mBounds.mMaxLon << "%29";
  }

  int32_t width  = coverage.getSettings().mAxisResolution[0];
  int32_t height = coverage.getSettings().mAxisResolution[1];

  if (request.mMaxSize > 0 && (width > request.mMaxSize || height > request.mMaxSize)) {
    double aspect = static_cast<double>(width) / static_cast<double>(height);
    width         = aspect > 1 ? request.mMaxSize : static_cast<int32_t>(request.mMaxSize * aspect);
    height        = aspect > 1 ? static_cast<int32_t>(request.mMaxSize / aspect) : request.mMaxSize;

    width  = std::max(1, width);  // Ensure width is at least 1
    height = std::max(1, height); // Ensure height is at least 1

    // &SCALESIZE=i(...),j(...)
    url << "&SCALESIZE=" << coverage.getSettings().mAxisLabels[0] << "%28" << width << "%29";
    url << "," << coverage.getSettings().mAxisLabels[1] << "%28" << height << "%29";
  }

  // Add time string to map server request if time is specified
  if (request.mTime.has_value()) {
    // &SUBSET=time("...")
    url << "&SUBSET=time%28%22" << request.mTime.value() << "%22%29";
  }

  // &SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326
  url << "&SUBSETTINGCRS=http%3A%2F%2Fwww.opengis.net%2Fdef%2Fcrs%2FEPSG%2F0%2F4326";

  url << "&FORMAT=" << request.mFormat.value_or("image%2Ftiff");

  if (request.mLayerRange.has_value()) {
    int minLayer = std::max(1, request.mLayerRange.value().first);
    int maxLayer = std::min(coverage.getSettings().mNumLayers, request.mLayerRange.value().second);

    if (minLayer == maxLayer) {
      // url << "&RANGESUBSET=" << minLayer;
    } else {
      url << "&RANGESUBSET=" << minLayer << "%3A" << maxLayer;
    }
  }

  return url.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::ogc