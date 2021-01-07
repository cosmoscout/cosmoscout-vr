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

std::future<std::string> WebMapTextureLoader::loadTextureAsync(WebMapService const& wms,
    WebMapLayer const& layer, std::string const& time, std::string const& mapCache,
    int const& maxSize, std::array<double, 2> lonRange, std::array<double, 2> latRange) {
  return mThreadPool.enqueue(
      [=]() { return loadTexture(wms, layer, time, mapCache, maxSize, lonRange, latRange); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::string> WebMapTextureLoader::loadTextureAsync(WebMapService const& wms,
    WebMapLayer const& layer, std::string const& time, std::string const& mapCache,
    int const& maxSize) {
  return mThreadPool.enqueue([=]() { return loadTexture(wms, layer, time, mapCache, maxSize); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapTextureLoader::loadTexture(WebMapService const& wms, WebMapLayer const& layer,
    std::string const& time, std::string const& mapCache, int const& maxSize,
    std::array<double, 2> lonRange, std::array<double, 2> latRange) {

  if (layer.getSettings().mNoSubsets) {
    lonRange = layer.getSettings().mLonRange;
    latRange = layer.getSettings().mLatRange;
  }

  std::string mime          = getMimeType();
  auto        cacheFilePath = getCachePath(layer, time, mapCache, lonRange, latRange, mime);

  // the file is already there, we can return it
  if (boost::filesystem::exists(cacheFilePath) && boost::filesystem::file_size(cacheFilePath) > 0) {
    return cacheFilePath.string();
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
        return "Error";
      }
    }
  }

  bool fail = false;
  {
    std::ofstream out;
    out.open(cacheFilePath.string(), std::ofstream::out | std::ofstream::binary);

    if (!out) {
      logger().error("Failed to open '{}' for writing!", cacheFilePath.string());
      return "Error";
    }

    std::string url = getRequestUrl(wms, layer, time, maxSize, lonRange, latRange, mime);

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
      return "Error";
    }

    // Check if the content type is correct.
    std::string contentType = curlpp::Info<CURLINFO_CONTENT_TYPE, std::string>::get(request);
    if (contentType == "NULL" || contentType != mime) {
      fail = true;
    }
  }

  if (fail) {
    logger().warn("There is no image to load for time {}.", time);
    boost::filesystem::remove(cacheFilePath);
    return "Error";
  }

  boost::filesystem::perms filePerms =
      boost::filesystem::perms::owner_read | boost::filesystem::perms::owner_write |
      boost::filesystem::perms::group_read | boost::filesystem::perms::group_write |
      boost::filesystem::perms::others_read | boost::filesystem::perms::others_write;
  boost::filesystem::permissions(cacheFilePath, filePerms);

  return cacheFilePath.string();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapTextureLoader::loadTexture(WebMapService const& wms, WebMapLayer const& layer,
    std::string const& time, std::string const& mapCache, int const& maxSize) {
  return loadTexture(wms, layer, time, mapCache, maxSize, layer.getSettings().mLonRange,
      layer.getSettings().mLatRange);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<WebMapTexture> WebMapTextureLoader::loadTextureFromFileAsync(
    std::string const& fileName) {
  return mThreadPool.enqueue([=]() {
    int width, height, bpp;
    int channels = 4;

    unsigned char* pixels = stbi_load(fileName.c_str(), &width, &height, &bpp, channels);

    if (!pixels) {
      logger().error("Failed to load '{}' with stbi!", fileName.c_str());
      pixels = reinterpret_cast<unsigned char*>(const_cast<char*>("Error"));
    }

    WebMapTexture texture{pixels, width, height};
    return texture;
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

boost::filesystem::path WebMapTextureLoader::getCachePath(WebMapLayer const& layer,
    std::string const& time, std::string const& mapCache, std::array<double, 2> const& lonRange,
    std::array<double, 2> const& latRange, std::string const& mime) {

  // Replace forbidden characters in layer string before creating cache dir.
  std::string layerFixed;
  boost::replace_copy_if(
      layer.getName(), std::back_inserter(layerFixed), boost::is_any_of("*.,:[|]\""), '_');

  // Set file format to three caracters.
  std::string fileFormat = mMimeToExtension.at(mime);

  std::stringstream cacheDir;
  cacheDir << mapCache << "/" << layerFixed << "/";
  cacheDir << lonRange[0] << "_" << latRange[0] << "_" << lonRange[1] << "_" << latRange[1] << "/";

  // Add year subdirectory, if time is specified.
  if (time != "") {
    std::string       year;
    std::stringstream time_stringstream(time);

    // Create dir for year.
    std::getline(time_stringstream, year, '-');
    cacheDir << year << "/";
  }

  std::stringstream cacheFile(cacheDir.str());

  // Add time string to cache file name if time is specified
  if (time != "") {
    std::string timeForFile = time;
    std::replace(timeForFile.begin(), timeForFile.end(), '/', '-');
    std::replace(timeForFile.begin(), timeForFile.end(), ':', '-');

    cacheFile << cacheDir.str() << timeForFile << "." << fileFormat;
  } else {
    cacheFile << cacheDir.str() << layerFixed << "." << fileFormat;
  }

  return boost::filesystem::path(cacheFile.str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapTextureLoader::getRequestUrl(WebMapService const& wms, WebMapLayer const& layer,
    std::string const& time, int const& maxSize, std::array<double, 2> const& lonRange,
    std::array<double, 2> const& latRange, std::string const& mime) {

  std::stringstream url;
  url.precision(std::numeric_limits<double>::max_digits10);

  url << wms.getUrl();
  url << "?SERVICE=WMS";
  url << "&VERSION=1.3.0";
  url << "&REQUEST=GetMap";
  url << "&FORMAT=" << mime;
  url << "&CRS=CRS:84";
  url << "&LAYERS=" << layer.getName();
  url << "&BBOX=" << lonRange[0] << "," << latRange[0] << "," << lonRange[1] << "," << latRange[1];

  if (layer.getSettings().mOpaque) {
    url << "&TRANSPARENT=FALSE";
  } else {
    url << "&TRANSPARENT=TRUE";
  }

  double             aspect = (lonRange[1] - lonRange[0]) / (latRange[1] - latRange[0]);
  std::optional<int> width, height;

  width  = layer.getSettings().mFixedWidth;
  height = layer.getSettings().mFixedWidth;

  if (!width.has_value() && !height.has_value()) {
    if (aspect < 1) {
      height = maxSize;
    } else {
      width = maxSize;
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
  if (time != "") {
    url << "&TIME=" << time;
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
