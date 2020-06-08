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
#include <boost/range/algorithm/replace_copy_if.hpp>
#include <curlpp/Infos.hpp>
#include <curlpp/Options.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

namespace csp::simplewmsbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapTextureLoader::WebMapTextureLoader()
    : mThreadPool(32) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapTextureLoader::~WebMapTextureLoader() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebMapTextureLoader::fileExist(const char* fileName) {
  std::ifstream infile(fileName);
  return infile.good();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapTextureLoader::loadTexture(std::string time, std::string requestStr,
    std::string const& layer, std::string const& mapCache) {

  // Replace forbidden characters in layer string before creating cache dir.
  std::string layerFixed;
  boost::replace_copy_if(layer, std::back_inserter(layerFixed), boost::is_any_of("*.,:[|]\""), '_');

  std::string cacheDir = mapCache + "/" + layerFixed + "/";

  // Add year subdirectory, if time is specified.
  if (time != "") {
    std::string       year;
    std::stringstream time_stringstream(time);

    // Create dir for year.
    std::getline(time_stringstream, year, '-');
    cacheDir += year + "/";
  }

  auto cacheDirPath(boost::filesystem::absolute(boost::filesystem::path(cacheDir)));

  if (!(boost::filesystem::exists(cacheDirPath))) {
    try {
      cs::utils::filesystem::createDirectoryRecursively(
          cacheDirPath, boost::filesystem::perms::all_all);
    } catch (std::exception& e) {
      logger().error("Failed to create cache directory: '{}'!", e.what());
      return "Error";
    }
  }

  std::string cacheFile;

  // Add time string to map server request if time is specified
  if (time != "") {
    requestStr += "&TIME=";
    requestStr += time;

    std::replace(time.begin(), time.end(), '/', '-');
    std::replace(time.begin(), time.end(), ':', '-');

    cacheFile = cacheDir + time + ".png";
  } else {
    cacheFile = cacheDir + layerFixed + ".png";
  }

  // No need to download the file if it is already in cache.
  if (fileExist(cacheFile.c_str())) {
    return cacheFile;
  }

  // Create cache file for write.
  std::ofstream out;
  out.open(cacheFile, std::ofstream::out | std::ofstream::binary);

  if (!out) {
    logger().error("Failed to open '{}' for writing!", cacheFile);
    return "Error";
  }

  curlpp::Easy request;
  request.setOpt(curlpp::options::Url(requestStr));
  request.setOpt(curlpp::options::WriteStream(&out));
  request.setOpt(curlpp::options::NoSignal(true));

  // Load to cache file.
  try {
    request.perform();
  } catch (std::exception& e) {
    logger().error("Failed to perform WMS request: '{}'! Exception: '{}'", requestStr, e.what());
    out.close();
    remove(cacheFile.c_str());
    return "Error";
  }

  bool fail =
      curlpp::Info<CURLINFO_CONTENT_TYPE, std::string>::get(request).substr(0, 9) != "image/png";

  if (fail) {
    logger().warn("There is no image to load for time {}.", time);
    out.close();
    remove(cacheFile.c_str());
    return "Error";
  }

  out.close();

  return cacheFile;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::string> WebMapTextureLoader::loadTextureAsync(std::string time,
    std::string requestStr, std::string const& layer, std::string const& mapCache) {
  return mThreadPool.enqueue([=]() { return loadTexture(time, requestStr, layer, mapCache); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<unsigned char*> WebMapTextureLoader::loadTextureFromFileAsync(
    std::string const& fileName) {
  return mThreadPool.enqueue([=]() {
    int width, height, bpp;
    int channels = 4;
    return stbi_load(fileName.c_str(), &width, &height, &bpp, channels);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::simplewmsbodies