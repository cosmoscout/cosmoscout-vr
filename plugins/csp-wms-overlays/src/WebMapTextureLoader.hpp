////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WMS_TEXTURE_LOADER_HPP
#define CSP_WMS_TEXTURE_LOADER_HPP

#include "WebMapLayer.hpp"
#include "WebMapService.hpp"

#include "../../../src/cs-utils/ThreadPool.hpp"

#include <boost/filesystem.hpp>

#include <array>
#include <map>

namespace csp::wmsoverlays {

struct WebMapTexture {
  unsigned char* mData;
  int            mWidth;
  int            mHeight;
  Bounds         mBounds;
};

class WebMapTextureLoader {
 public:
  struct Request {
    int                        mMaxSize;
    std::string                mStyle;
    std::optional<std::string> mTime;
    std::optional<Bounds>      mBounds;
  };

  /// Create a new ThreadPool with the specified amount of threads.
  WebMapTextureLoader();

  ~WebMapTextureLoader();

  /// Async WMS texture loader.
  std::future<std::optional<WebMapTexture>> loadTextureAsync(WebMapService const& wms,
      WebMapLayer const& layer, Request const& request, std::string const& mapCache);

  /// WMS texture loader.
  std::optional<WebMapTexture> loadTexture(WebMapService const& wms, WebMapLayer const& layer,
      Request request, std::string const& mapCache);

 private:
  std::optional<std::stringstream> requestTexture(WebMapService const& wms,
      WebMapLayer const& layer, Request const& request, std::string const& mapCache);

  void saveTextureToFile(boost::filesystem::path const& file, std::stringstream const& data);

  /// Load WMS texture from a file using stbi.
  std::optional<WebMapTexture> loadTextureFromFile(std::string const& fileName);

  /// Load WMS texture from a stream using stbi.
  std::optional<WebMapTexture> loadTextureFromStream(std::stringstream const& stream);

  boost::filesystem::path getCachePath(WebMapService const& wms, WebMapLayer const& layer,
      Request const& request, std::string const& mapCache);
  std::string             getRequestUrl(
                  WebMapService const& wms, WebMapLayer const& layer, Request const& request);
  std::string getMimeType(WebMapService const& wms, WebMapLayer const& layer);

  const std::map<std::string, std::string> mMimeToExtension = {
      {"image/png", "png"}, {"image/jpeg", "jpg"}};

  std::mutex            mTextureMutex;
  cs::utils::ThreadPool mThreadPool;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_TEXTURE_LOADER_HPP
