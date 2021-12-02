////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebCoverageTextureLoader.hpp"

#include "WebCoverageException.hpp"

#include "../../../src/cs-utils/filesystem.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm/replace_copy_if.hpp>
#include <curlpp/Infos.hpp>
#include <curlpp/Options.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace csp::wcsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverageTextureLoader::WebCoverageTextureLoader()
    : mThreadPool(32) {
  GDALReader::InitGDAL();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::optional<GDALReader::GreyScaleTexture>> WebCoverageTextureLoader::loadTextureAsync(
    WebCoverageService const& wcs, WebCoverage const& coverage, Request const& request,
    std::string const& coverageCache, bool saveToCache) {
  return mThreadPool.enqueue(
      [=]() { return loadTexture(wcs, coverage, request, coverageCache, saveToCache); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<GDALReader::GreyScaleTexture> WebCoverageTextureLoader::loadTexture(
    WebCoverageService const& wcs, WebCoverage const& coverage, Request const& request,
    std::string const& coverageCache, bool saveToCache) {
  boost::filesystem::path cachePath = getCachePath(wcs, coverage, request, coverageCache);

  std::optional<std::stringstream> textureStream;
  GDALReader::GreyScaleTexture     texture;

  if (saveToCache && boost::filesystem::exists(cachePath) &&
      boost::filesystem::file_size(cachePath) > 0) {
    GDALReader::ReadGrayScaleTexture(texture, cachePath.string(), request.layer.value_or(1));
  } else {
    textureStream = requestTexture(wcs, coverage, request);
    if (!textureStream.has_value()) {
      return {};
    }

    GDALReader::ReadGrayScaleTexture(
        texture, textureStream.value(), cachePath.string(), request.layer.value_or(1));

    if (saveToCache) {
      textureStream.value().rdbuf()->pubseekpos(0);
      saveTextureToFile(cachePath, textureStream.value());
    }
  }

  if (!texture.buffer) {
    return {};
  }

  return {texture};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::stringstream> WebCoverageTextureLoader::requestTexture(
    WebCoverageService const& wcs, WebCoverage const& layer, Request const& wcsrequest) {

  std::string url = getRequestUrl(wcs, layer, wcsrequest);

  logger().debug("Performing WCS request '{}'.", url);

  int maxRetries = 3;
  for (int i = 0; i < maxRetries; i++) {
    if (i > 0) {
      logger().debug("Retrying...");
    }

    std::stringstream out(std::ios_base::out | std::ios_base::in | std::ios_base::binary);

    curlpp::Easy request;
    request.setOpt(curlpp::options::Url(url));
    request.setOpt(curlpp::options::WriteStream(&out));
    request.setOpt(curlpp::options::NoSignal(true));
    request.setOpt(curlpp::options::SslVerifyPeer(false));
    request.setOpt(curlpp::options::FollowLocation(true));

    try {
      request.perform();
    } catch (std::exception& e) {
      logger().warn("Failed to perform WCS request '{}': '{}'!", url, e.what());
      continue;
    }

    std::string contentType = curlpp::Info<CURLINFO_CONTENT_TYPE, std::string>::get(request);
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
    } else if (contentType != wcsrequest.mFormat.value_or("image/tiff")) {
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
  std::string layerFixed;
  boost::replace_copy_if(
      coverage.getId(), std::back_inserter(layerFixed), boost::is_any_of("*.,:[|]\""), '_');

  // Set file format to three characters.
  std::string fileFormat = mMimeToExtension.at(request.mFormat.value_or("image/tiff"));

  std::stringstream cacheDir;
  cacheDir << coverageCache << "/" << layerFixed << "/";
  cacheDir << request.mMaxSize << "px/";

  if (request.mTime.has_value()) {
    std::string       year;
    std::stringstream time_stringstream(request.mTime.value());

    // Create dir for year.
    std::getline(time_stringstream, year, '-');
    cacheDir << year << "/";
  }

  std::stringstream cacheFile(cacheDir.str());

  std::stringstream layer;
  layer << "_Layer_" << std::to_string(request.layer.value_or(1));

  // Add time string to cache file name if time is specified
  if (request.mTime.has_value()) {
    std::string timeForFile = request.mTime.value();
    std::replace(timeForFile.begin(), timeForFile.end(), '/', '-');
    std::replace(timeForFile.begin(), timeForFile.end(), ':', '-');

    cacheFile << cacheDir.str() << timeForFile << layer.str() << "." << fileFormat;
  } else {
    cacheFile << cacheDir.str() << layerFixed << layer.str() << "." << fileFormat;
  }

  return boost::filesystem::path(cacheFile.str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebCoverageTextureLoader::getRequestUrl(
    WebCoverageService const& wcs, WebCoverage const& coverage, Request const& request) {

  std::stringstream url;
  url.precision(std::numeric_limits<double>::max_digits10);

  url << wcs.getUrl();
  url << "?SERVICE=WCS";
  url << "&VERSION=2.0.1";
  url << "&REQUEST=GetCoverage";
  url << "&COVERAGEID=" << coverage.getId();

  /// All special chars need to be in url encoded form
  /// This is only really an issue with tomcat servers

  if (request.mBounds != coverage.getSettings().mBounds && request.mBounds != Bounds()) {
    // &SUBSET=Lat(...,...)
    url << "&SUBSET=Lat%28" << request.mBounds.mMinLat << "," << request.mBounds.mMaxLat << "%29";
    // &SUBSET=Long(...,...)
    url << "&SUBSET=Long%28" << request.mBounds.mMinLon << "," << request.mBounds.mMaxLon << "%29";
  }

  if (request.mMaxSize > 0 && coverage.getSettings().mAxisLabels.size() == 2) {
    // &SCALESIZE=i(...),j(...)
    url << "&SCALESIZE=" << coverage.getSettings().mAxisLabels[0] << "%28" << request.mMaxSize
        << "%29";
    url << "," << coverage.getSettings().mAxisLabels[1] << "%28" << request.mMaxSize << "%29";
  }

  // Add time string to map server request if time is specified
  if (request.mTime.has_value()) {
    // &SUBSET=time("...")
    url << "&SUBSET=time%28%22" << request.mTime.value() << "%22%29";
  }

  // &SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326
  url << "&SUBSETTINGCRS=http%3A%2F%2Fwww.opengis.net%2Fdef%2Fcrs%2FEPSG%2F0%2F4326";

  url << "&FORMAT=" << request.mFormat.value_or("image%2Ftiff");

  return url.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wcsoverlays
