////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebMapTextureLoader.hpp"

#include "WebMapException.hpp"

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

std::future<std::optional<WebMapTexture>> WebMapTextureLoader::loadTextureAsync(
    WebMapService const& wms, WebMapLayer const& layer, Request const& request,
    std::string const& mapCache, bool saveToCache) {
  return mThreadPool.enqueue(
      [=]() { return loadTexture(wms, layer, request, mapCache, saveToCache); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<WebMapTexture> WebMapTextureLoader::loadTexture(WebMapService const& wms,
    WebMapLayer const& layer, Request const& request, std::string const& mapCache,
    bool saveToCache) {
  boost::filesystem::path cachePath = getCachePath(wms, layer, request, mapCache);
  if (saveToCache) {
    // The file is already there, we can return it
    if (boost::filesystem::exists(cachePath) && boost::filesystem::file_size(cachePath) > 0) {
      std::optional<WebMapTexture> texture = loadTextureFromFile(cachePath.string());
      return texture;
    }
  }

  // The file is corrupt or not available, we have to request it
  auto textureStream = requestTexture(wms, layer, request);
  if (!textureStream.has_value()) {
    return {};
  }

  if (saveToCache) {
    saveTextureToFile(cachePath, textureStream.value());
  }

  std::optional<WebMapTexture> texture = loadTextureFromStream(textureStream.value());
  return texture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::stringstream> WebMapTextureLoader::requestTexture(
    WebMapService const& wms, WebMapLayer const& layer, Request const& wmsrequest) {

  std::string url = getRequestUrl(wms, layer, wmsrequest);

  logger().debug("Performing WMS request '{}'.", url);

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

    try {
      request.perform();
    } catch (std::exception& e) {
      logger().warn("Failed to perform WMS request '{}': '{}'!", url, e.what());
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
    if (contentType == "text/xml") {
      // A WMS exception might have occurred.
      try {
        // If there was a valid WMS exception, the problem probably can't be fixed with a retry.
        // => Return an empty object to cancel the request.
        WebMapExceptionReport e(out.str());
        logger().warn("WMS Exception occurred for WMS request '{}': '{}'!", url, e.what());
        return {};
      } catch (std::exception const& e) {
        // If parsing the document fails, this might be due to connection problems
        // or corrupted data.
        // => Retry the request.
        logger().debug("Could not create WebMapExceptionReport: '{}'.", e.what());
        continue;
      }
      continue;
    } else if (contentType != getMimeType(wms, layer)) {
      logger().debug("Received response of invalid MIME type '{}'.", contentType);
      continue;
    }
    return std::move(out);
  }
  logger().warn("Could not get a valid response for WMS request '{}'!", url);
  return {};
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

std::optional<WebMapTexture> WebMapTextureLoader::loadTextureFromFile(std::string const& fileName) {
  int width, height, bpp;
  int channels = 4;

  std::unique_ptr<unsigned char> pixels(
      stbi_load(fileName.c_str(), &width, &height, &bpp, channels));

  if (!pixels) {
    logger().warn("Failed to load '{}' with stbi!", fileName);
    return std::optional<WebMapTexture>{};
  }

  WebMapTexture texture{std::move(pixels), width, height};
  return std::move(texture);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<WebMapTexture> WebMapTextureLoader::loadTextureFromStream(
    std::stringstream const& stream) {
  int width, height, bpp;
  int channels = 4;

  std::unique_ptr<unsigned char> pixels(
      stbi_load_from_memory(reinterpret_cast<unsigned char*>(stream.str().data()),
          static_cast<int>(stream.str().size()), &width, &height, &bpp, channels));

  if (!pixels) {
    logger().warn("Failed to load texture from memory with stbi!");
    return std::optional<WebMapTexture>{};
  }

  WebMapTexture texture{std::move(pixels), width, height};
  return std::move(texture);
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
  cacheDir << request.mMaxSize << "px/";

  // Add year subdirectory, if time is specified.
  if (request.mTime.has_value()) {
    std::string       year;
    std::stringstream time_stringstream(request.mTime.value());

    // Create dir for year.
    std::getline(time_stringstream, year, '-');
    cacheDir << year << "/";
  }

  if (!request.mStyle.empty()) {
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
    url << "&BBOX=" << request.mBounds.mMinLon << "," << request.mBounds.mMinLat << ","
        << request.mBounds.mMaxLon << "," << request.mBounds.mMaxLat;
  } else if (cs::utils::contains(layer.getSettings().mCrs, "EPSG:4326")) {
    url << "&CRS=EPSG:4326";
    url << "&BBOX=" << request.mBounds.mMinLat << "," << request.mBounds.mMinLon << ","
        << request.mBounds.mMaxLat << "," << request.mBounds.mMaxLon;
  } else {
    logger().warn("No compatible CRS found. Trying CRS:84 anyway");
    url << "&CRS=CRS:84";
    url << "&BBOX=" << request.mBounds.mMinLon << "," << request.mBounds.mMinLat << ","
        << request.mBounds.mMaxLon << "," << request.mBounds.mMaxLat;
  }

  if (layer.getSettings().mOpaque) {
    url << "&TRANSPARENT=FALSE";
  } else {
    url << "&TRANSPARENT=TRUE";
  }

  double aspect = (request.mBounds.mMaxLon - request.mBounds.mMinLon) /
                  (request.mBounds.mMaxLat - request.mBounds.mMinLat);
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
    height = static_cast<int>(static_cast<double>(width.value()) / aspect);
  } else if (height.has_value() && !width.has_value()) {
    width = static_cast<int>(static_cast<double>(height.value()) * aspect);
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
