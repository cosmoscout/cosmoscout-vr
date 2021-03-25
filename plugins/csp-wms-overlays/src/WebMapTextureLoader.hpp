////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WMS_OVERLAYS_TEXTURE_LOADER_HPP
#define CSP_WMS_OVERLAYS_TEXTURE_LOADER_HPP

#include "WebMapLayer.hpp"
#include "WebMapService.hpp"

#include "../../../src/cs-utils/ThreadPool.hpp"

#include <boost/filesystem.hpp>

#include <array>
#include <map>

namespace csp::wmsoverlays {

/// Struct for storing texture data along with some metadata.
struct WebMapTexture {
  std::unique_ptr<unsigned char> mData;
  int                            mWidth;
  int                            mHeight;
};

/// Class for requesting map textures from Web Map Services.
class WebMapTextureLoader {
 public:
  /// Struct for defining parameters for a request to a WMS.
  struct Request {
    int                        mMaxSize{};
    std::string                mStyle;
    Bounds                     mBounds;
    std::optional<std::string> mTime;
  };

  /// Creates a new ThreadPool with the specified amount of threads.
  WebMapTextureLoader();

  /// Async WMS texture loader.
  /// Returns an empty optional if loading the texture failed.
  std::future<std::optional<WebMapTexture>> loadTextureAsync(WebMapService const& wms,
      WebMapLayer const& layer, Request const& request, std::string const& mapCache,
      bool saveToCache);

  /// WMS texture loader.
  /// Returns an empty optional if loading the texture failed.
  std::optional<WebMapTexture> loadTexture(WebMapService const& wms, WebMapLayer const& layer,
      Request const& request, std::string const& mapCache, bool saveToCache);

 private:
  /// Requests a map texture from a WMS.
  /// Returns a binary stream of the texture file if the request succeeds.
  /// Returns an empty optional if the request fails.
  std::optional<std::stringstream> requestTexture(
      WebMapService const& wms, WebMapLayer const& layer, Request const& request);

  /// Saves a binary stream of a texture file to the given path.
  void saveTextureToFile(boost::filesystem::path const& file, std::stringstream const& data);

  /// Loads WMS texture from a file using stbi.
  static std::optional<WebMapTexture> loadTextureFromFile(std::string const& fileName);

  /// Loads WMS texture from a stream using stbi.
  static std::optional<WebMapTexture> loadTextureFromStream(std::stringstream const& stream);

  /// Constructs a path for loading/saving the texture requested with the given parameters.
  boost::filesystem::path getCachePath(WebMapService const& wms, WebMapLayer const& layer,
      Request const& request, std::string const& mapCache);

  /// Constructs a request URL for the given parameters.
  std::string getRequestUrl(
      WebMapService const& wms, WebMapLayer const& layer, Request const& request);

  /// Determines an appropriate MIME type for requesting a texture for the given layer.
  static std::string getMimeType(WebMapService const& wms, WebMapLayer const& layer);

  const std::map<std::string, std::string> mMimeToExtension = {
      {"image/png", "png"}, {"image/jpeg", "jpg"}};

  std::mutex            mTextureMutex;
  cs::utils::ThreadPool mThreadPool;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_OVERLAYS_TEXTURE_LOADER_HPP
