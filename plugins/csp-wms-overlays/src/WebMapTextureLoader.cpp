////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebMapTextureLoader.hpp"

#include "logger.hpp"

#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

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

std::future<std::optional<WebMapTexture>> WebMapTextureLoader::loadTextureAsync(
    WebMapService const& wms, WebMapLayer const& layer, Request const& request,
    std::string const& mapCache) {
  return mThreadPool.enqueue([=]() { return loadTexture(wms, layer, request, mapCache); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<WebMapTexture> WebMapTextureLoader::loadTexture(WebMapService const& wms,
    WebMapLayer const& layer, Request request, std::string const& mapCache) {
  if (layer.getSettings().mNoSubsets) {
    request.mBounds = layer.getSettings().mBounds;
  } else {
    request.mBounds = request.mBounds.value_or(layer.getSettings().mBounds);
  }

  boost::filesystem::path cachePath = getCachePath(wms, layer, request, mapCache);

  // The file is already there, we can return it
  if (boost::filesystem::exists(cachePath) && boost::filesystem::file_size(cachePath) > 0) {
    std::optional<WebMapTexture> texture = loadTextureFromFile(cachePath.string());
    if (texture.has_value()) {
      texture->mBounds = request.mBounds.value();
    }
    return texture;
  }

  // The file is corrupt or not available, we have to request it
  auto textureStream = requestTexture(wms, layer, request, mapCache);
  if (!textureStream.has_value()) {
    return {};
  }

  saveTextureToFile(cachePath, textureStream.value());

  std::optional<WebMapTexture> texture = loadTextureFromStream(textureStream.value());
  if (texture.has_value()) {
    texture->mBounds = request.mBounds.value();
  }
  return texture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::stringstream> WebMapTextureLoader::requestTexture(WebMapService const& wms,
    WebMapLayer const& layer, Request const& wmsrequest, std::string const& mapCache) {
  std::stringstream out(std::ios_base::out | std::ios_base::in | std::ios_base::binary);

  std::string url = getRequestUrl(wms, layer, wmsrequest);

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
    return {};
  }

  // Check if the content type is correct.
  std::string contentType = curlpp::Info<CURLINFO_CONTENT_TYPE, std::string>::get(request);
  if (contentType == "NULL" || contentType != getMimeType(wms, layer)) {
    if (wmsrequest.mTime.has_value()) {
      logger().warn("There is no image to load for layer '{}' at time {}.", layer.getTitle(),
          wmsrequest.mTime.value());
    } else {
      logger().warn("There is no image to load for layer '{}'.", layer.getTitle());
    }
    return {};
  }

  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebMapTextureLoader::saveTextureToFile(
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
        logger().error("Failed to create cache directory: {}", e.what());
        return;
      }
    }
  }

  {
    std::ofstream out;
    out.open(file.string(), std::ofstream::out | std::ofstream::binary);

    if (!out) {
      logger().error("Failed to open '{}' for writing!", file.string());
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

std::optional<WebMapTexture> WebMapTextureLoader::loadTextureFromFile(std::string const& fileName) {
  int width, height, bpp;
  int channels = 4;

  unsigned char* pixels = stbi_load(fileName.c_str(), &width, &height, &bpp, channels);

  if (!pixels) {
    logger().error("Failed to load '{}' with stbi!", fileName.c_str());
    return std::optional<WebMapTexture>{};
  }

  WebMapTexture texture{pixels, width, height};
  return std::optional<WebMapTexture>{texture};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<WebMapTexture> WebMapTextureLoader::loadTextureFromStream(
    std::stringstream const& stream) {
  int width, height, bpp;
  int channels = 4;

  unsigned char* pixels = stbi_load_from_memory((unsigned char*)stream.str().data(),
      (int)stream.str().size(), &width, &height, &bpp, channels);

  if (!pixels) {
    logger().error("Failed to load texture from memory with stbi!");
    return std::optional<WebMapTexture>{};
  }

  WebMapTexture texture{pixels, width, height};
  return std::optional<WebMapTexture>{texture};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

boost::filesystem::path WebMapTextureLoader::getCachePath(WebMapService const& wms,
    WebMapLayer const& layer, Request const& request, std::string const& mapCache) {

  // Replace forbidden characters in layer string before creating cache dir.
  std::string layerFixed;
  boost::replace_copy_if(
      layer.getName(), std::back_inserter(layerFixed), boost::is_any_of("*.,:[|]\""), '_');

  // Set file format to three caracters.
  std::string fileFormat = mMimeToExtension.at(getMimeType(wms, layer));

  std::stringstream cacheDir;
  cacheDir << mapCache << "/" << layerFixed << "/";
  cacheDir << request.mBounds.value().mMinLon << "_" << request.mBounds.value().mMinLat << "_"
           << request.mBounds.value().mMaxLon << "_" << request.mBounds.value().mMaxLat << "/";

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
  url << "&FORMAT=" << getMimeType(wms, layer);
  url << "&LAYERS=" << layer.getName();
  url << "&STYLES=" << request.mStyle;

  if (cs::utils::contains(layer.getSettings().mCrs, "CRS:84")) {
    url << "&CRS=CRS:84";
    url << "&BBOX=" << request.mBounds.value().mMinLon << "," << request.mBounds.value().mMinLat
        << "," << request.mBounds.value().mMaxLon << "," << request.mBounds.value().mMaxLat;
  } else if (cs::utils::contains(layer.getSettings().mCrs, "EPSG:4326")) {
    url << "&CRS=EPSG:4326";
    url << "&BBOX=" << request.mBounds.value().mMinLat << "," << request.mBounds.value().mMinLon
        << "," << request.mBounds.value().mMaxLat << "," << request.mBounds.value().mMaxLon;
  } else {
    logger().warn("No compatible CRS found. Trying CRS:84 anyway");
    url << "&CRS=CRS:84";
    url << "&BBOX=" << request.mBounds.value().mMinLon << "," << request.mBounds.value().mMinLat
        << "," << request.mBounds.value().mMaxLon << "," << request.mBounds.value().mMaxLat;
  }

  if (layer.getSettings().mOpaque) {
    url << "&TRANSPARENT=FALSE";
  } else {
    url << "&TRANSPARENT=TRUE";
  }

  double aspect = (request.mBounds.value().mMaxLon - request.mBounds.value().mMinLon) /
                  (request.mBounds.value().mMaxLat - request.mBounds.value().mMinLat);
  std::optional<int> width, height;

  width  = layer.getSettings().mFixedWidth;
  height = layer.getSettings().mFixedHeight;

  if (!width.has_value() && !height.has_value()) {
    if (aspect < 1) {
      height = std::min(
          request.mMaxSize, wms.getSettings().mMaxHeight.value_or(std::numeric_limits<int>::max()));
    } else {
      width = std::min(
          request.mMaxSize, wms.getSettings().mMaxWidth.value_or(std::numeric_limits<int>::max()));
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

std::string WebMapTextureLoader::getMimeType(WebMapService const& wms, WebMapLayer const& layer) {
  if (layer.getSettings().mOpaque) {
    if (wms.isFormatSupported("image/jpeg")) {
      return "image/png";
    }
  }
  // TODO Use other format if png is not supported
  return "image/png";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
