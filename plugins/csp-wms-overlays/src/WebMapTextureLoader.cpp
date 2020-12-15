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

std::string WebMapTextureLoader::loadTexture(std::string time, std::string const& requestStr,
    std::string const& format, std::string const& layer, std::array<double, 2> const& latRange,
    std::array<double, 2> const& lonRange, std::string const& mapCache) {

  // Replace forbidden characters in layer string before creating cache dir.
  std::string layerFixed;
  boost::replace_copy_if(layer, std::back_inserter(layerFixed), boost::is_any_of("*.,:[|]\""), '_');

  // Set file format to three caracters.
  std::string fileFormat = format;
  if (format == "jpeg") {
    fileFormat = "jpg";
  }

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
  std::stringstream url;

  url.precision(std::numeric_limits<double>::max_digits10);
  url << requestStr;

  // Add time string to map server request if time is specified
  if (time != "") {
    url << "&TIME=" << time << "&FORMAT=image/" << format;

    std::replace(time.begin(), time.end(), '/', '-');
    std::replace(time.begin(), time.end(), ':', '-');

    cacheFile << cacheDir.str() << time << "." << fileFormat;
  } else {
    cacheFile << cacheDir.str() << layerFixed << "." << fileFormat;
  }

  url << "&BBOX=" << lonRange[0] << "," << latRange[0] << "," << lonRange[1] << "," << latRange[1];

  auto cacheFilePath(boost::filesystem::path(cacheFile.str()));

  // the file is already there, we can return it
  if (boost::filesystem::exists(cacheFilePath) &&
      boost::filesystem::file_size(cacheFile.str()) > 0) {
    return cacheFile.str();
  }

  // the file is corrupt not available
  {
    std::unique_lock<std::mutex> lock(mTextureMutex);

    if (boost::filesystem::exists(cacheFilePath) &&
        boost::filesystem::file_size(cacheFile.str()) == 0) {
      boost::filesystem::remove(cacheFilePath);
    }

    auto cacheDirPath(boost::filesystem::absolute(boost::filesystem::path(cacheDir.str())));
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
    out.open(cacheFile.str(), std::ofstream::out | std::ofstream::binary);

    if (!out) {
      logger().error("Failed to open '{}' for writing!", cacheFile.str());
      return "Error";
    }

    curlpp::Easy request;
    request.setOpt(curlpp::options::Url(url.str()));
    request.setOpt(curlpp::options::WriteStream(&out));
    request.setOpt(curlpp::options::NoSignal(true));
    request.setOpt(curlpp::options::SslVerifyPeer(false));

    logger().trace("URL: {}", url.str());

    // Load to cache file.
    try {
      request.perform();
    } catch (std::exception& e) {
      logger().error("Failed to perform WMS request: '{}'! Exception: '{}'", url.str(), e.what());
      std::remove(cacheFile.str().c_str());
      return "Error";
    }

    // Check if the content type is correct.
    std::string contentType = curlpp::Info<CURLINFO_CONTENT_TYPE, std::string>::get(request);
    if (contentType == "NULL" || contentType.substr(0, 6 + format.length()) != "image/" + format) {
      fail = true;
    }
  }

  if (fail) {
    logger().warn("There is no image to load for time {}.", time);
    std::remove(cacheFile.str().c_str());
    return "Error";
  }

  boost::filesystem::perms filePerms =
      boost::filesystem::perms::owner_read | boost::filesystem::perms::owner_write |
      boost::filesystem::perms::group_read | boost::filesystem::perms::group_write |
      boost::filesystem::perms::others_read | boost::filesystem::perms::others_write;
  boost::filesystem::permissions(cacheFilePath, filePerms);

  return cacheFile.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::string> WebMapTextureLoader::loadTextureAsync(std::string time,
    std::string const& requestStr, std::string const& format, std::string const& layer,
    std::array<double, 2> const& latRange, std::array<double, 2> const& lonRange,
    std::string const& mapCache) {
  return mThreadPool.enqueue(
      [=]() { return loadTexture(time, requestStr, format, layer, latRange, lonRange, mapCache); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<unsigned char*> WebMapTextureLoader::loadTextureFromFileAsync(
    std::string const& fileName) {
  return mThreadPool.enqueue([=]() {
    int width, height, bpp;
    int channels = 4;

    unsigned char* pixels = stbi_load(fileName.c_str(), &width, &height, &bpp, channels);

    if (!pixels) {
      logger().error("Failed to load '{}' with stbi!", fileName.c_str());
      pixels = reinterpret_cast<unsigned char*>(const_cast<char*>("Error"));
    }

    return pixels;
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
