////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebMapTextureLoader.hpp"

#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "logger.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/algorithm/replace_copy_if.hpp>
#include <curlpp/Infos.hpp>
#include <curlpp/Options.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

namespace csp::wmsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapTextureLoader::WebMapTextureLoader()
    : mThreadPool(32) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapTextureLoader::~WebMapTextureLoader() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::optional<WebMapTextureFile>> WebMapTextureLoader::loadTextureAsync(
    WebMapService const& wms, WebMapLayer const& layer, Request const& request,
    std::string const& mapCache) {
  return mThreadPool.enqueue([=]() { return loadTexture(wms, layer, request, mapCache); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<WebMapTextureFile> WebMapTextureLoader::loadTexture(WebMapService const& wms,
    WebMapLayer const& layer, Request request, std::string const& mapCache) {

  if (layer.getSettings().mNoSubsets) {
    request.mLonRange = layer.getSettings().mLonRange;
    request.mLatRange = layer.getSettings().mLatRange;
  } else {
    request.mLonRange = request.mLonRange.value_or(layer.getSettings().mLonRange);
    request.mLatRange = request.mLatRange.value_or(layer.getSettings().mLatRange);
  }

  WebMapTextureFile result;
  result.mLonRange = request.mLonRange.value();
  result.mLatRange = request.mLatRange.value();

  auto cacheFilePath = getCachePath(layer, request, mapCache);

  // the file is already there, we can return it
  if (boost::filesystem::exists(cacheFilePath) && boost::filesystem::file_size(cacheFilePath) > 0) {
    result.mPath = cacheFilePath.string();
    return result;
  }

  // the file is corrupt not available
  {
    std::unique_lock<std::mutex> lock(mTextureMutex);

    if (boost::filesystem::exists(cacheFilePath) &&
        boost::filesystem::file_size(cacheFilePath) == 0) {
      boost::filesystem::remove(cacheFilePath);
    }

    auto cacheDirPath(boost::filesystem::absolute(cacheFilePath.branch_path()));
    if (!(boost::filesystem::exists(cacheDirPath))) {
      try {
        cs::utils::filesystem::createDirectoryRecursively(
            cacheDirPath, boost::filesystem::perms::all_all);
      } catch (std::exception& e) {
        logger().error("Failed to create cache directory: {}", e.what());
        return {};
      }
    }
  }

  bool fail = false;
  {
    std::ofstream out;
    out.open(cacheFilePath.string(), std::ofstream::out | std::ofstream::binary);

    if (!out) {
      logger().error("Failed to open '{}' for writing!", cacheFilePath.string());
      return {};
    }

    std::string url = getRequestUrl(wms, layer, request);

    curlpp::Easy request;
    request.setOpt(curlpp::options::Url(url));
    request.setOpt(curlpp::options::WriteStream(&out));
    request.setOpt(curlpp::options::NoSignal(true));
    request.setOpt(curlpp::options::SslVerifyPeer(false));

    logger().trace("URL: {}", url);

    // Load to cache file.
    try {
      request.perform();
    } catch (std::exception& e) {
      logger().error("Failed to perform WMS request: '{}'! Exception: '{}'", url, e.what());
      boost::filesystem::remove(cacheFilePath);
      return {};
    }

    // Check if the content type is correct.
    std::string contentType = curlpp::Info<CURLINFO_CONTENT_TYPE, std::string>::get(request);
    if (contentType == "NULL" || contentType != getMimeType()) {
      fail = true;
    }
  }

  if (fail) {
    if (request.mTime.has_value()) {
      logger().warn("There is no image to load for layer '{}' at time {}.", layer.getTitle(),
          request.mTime.value());
    } else {
      logger().warn("There is no image to load for layer '{}'.", layer.getTitle());
    }
    boost::filesystem::remove(cacheFilePath);
    return {};
  }

  boost::filesystem::perms filePerms =
      boost::filesystem::perms::owner_read | boost::filesystem::perms::owner_write |
      boost::filesystem::perms::group_read | boost::filesystem::perms::group_write |
      boost::filesystem::perms::others_read | boost::filesystem::perms::others_write;
  boost::filesystem::permissions(cacheFilePath, filePerms);

  result.mPath = cacheFilePath.string();
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::optional<WebMapTexture>> WebMapTextureLoader::loadTextureFromFileAsync(
    std::string const& fileName) {
  return mThreadPool.enqueue([=]() {
    int width, height, bpp;
    int channels = 4;

    unsigned char* pixels = stbi_load(fileName.c_str(), &width, &height, &bpp, channels);

    if (!pixels) {
      logger().error("Failed to load '{}' with stbi!", fileName.c_str());
      return std::optional<WebMapTexture>{};
    }

    WebMapTexture texture{pixels, width, height};
    return std::optional<WebMapTexture>{texture};
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

boost::filesystem::path WebMapTextureLoader::getCachePath(
    WebMapLayer const& layer, Request const& request, std::string const& mapCache) {

  // Replace forbidden characters in layer string before creating cache dir.
  std::string layerFixed;
  boost::replace_copy_if(
      layer.getName(), std::back_inserter(layerFixed), boost::is_any_of("*.,:[|]\""), '_');

  // Set file format to three caracters.
  std::string fileFormat = mMimeToExtension.at(getMimeType());

  std::stringstream cacheDir;
  cacheDir << mapCache << "/" << layerFixed << "/";
  cacheDir << request.mLonRange.value()[0] << "_" << request.mLatRange.value()[0] << "_"
           << request.mLonRange.value()[1] << "_" << request.mLatRange.value()[1] << "/";

  // Add year subdirectory, if time is specified.
  if (request.mTime.has_value()) {
    std::string       year;
    std::stringstream time_stringstream(request.mTime.value());

    // Create dir for year.
    std::getline(time_stringstream, year, '-');
    cacheDir << year << "/";
  }

  if (request.mStyle != "") {
    cacheDir << request.mStyle << "/";
  }

  std::stringstream cacheFile(cacheDir.str());

  // Add time string to cache file name if time is specified
  if (request.mTime.has_value()) {
    std::string timeForFile = request.mTime.value();
    std::replace(timeForFile.begin(), timeForFile.end(), '/', '-');
    std::replace(timeForFile.begin(), timeForFile.end(), ':', '-');

    cacheFile << cacheDir.str() << timeForFile << "." << fileFormat;
  } else {
    cacheFile << cacheDir.str() << layerFixed << "." << fileFormat;
  }

  return boost::filesystem::path(cacheFile.str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapTextureLoader::getRequestUrl(
    WebMapService const& wms, WebMapLayer const& layer, Request const& request) {

  std::stringstream url;
  url.precision(std::numeric_limits<double>::max_digits10);

  url << wms.getUrl();
  url << "?SERVICE=WMS";
  url << "&VERSION=1.3.0";
  url << "&REQUEST=GetMap";
  url << "&FORMAT=" << getMimeType();
  url << "&CRS=CRS:84";
  url << "&LAYERS=" << layer.getName();
  url << "&STYLES=" << request.mStyle;
  url << "&BBOX=" << request.mLonRange.value()[0] << "," << request.mLatRange.value()[0] << ","
      << request.mLonRange.value()[1] << "," << request.mLatRange.value()[1];

  if (layer.getSettings().mOpaque) {
    url << "&TRANSPARENT=FALSE";
  } else {
    url << "&TRANSPARENT=TRUE";
  }

  double aspect = (request.mLonRange.value()[1] - request.mLonRange.value()[0]) /
                  (request.mLatRange.value()[1] - request.mLatRange.value()[0]);
  std::optional<int> width, height;

  width  = layer.getSettings().mFixedWidth;
  height = layer.getSettings().mFixedHeight;

  if (!width.has_value() && !height.has_value()) {
    if (aspect < 1) {
      height = request.mMaxSize;
    } else {
      width = request.mMaxSize;
    }
  }

  if (width.has_value() && !height.has_value()) {
    height = (int)((double)width.value() / aspect);
  } else if (height.has_value() && !width.has_value()) {
    width = (int)((double)height.value() * aspect);
  }

  url << "&WIDTH=" << width.value();
  url << "&HEIGHT=" << height.value();

  // Add time string to map server request if time is specified
  if (request.mTime.has_value()) {
    url << "&TIME=" << request.mTime.value();
  }

  return url.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapTextureLoader::getMimeType() {
  // TODO Better format handling
  return "image/png";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
