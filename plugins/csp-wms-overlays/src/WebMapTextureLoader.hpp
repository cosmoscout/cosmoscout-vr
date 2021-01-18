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

struct WebMapTextureFile {
  std::string           mPath;
  std::array<double, 2> mLonRange;
  std::array<double, 2> mLatRange;
};

struct WebMapTexture {
  unsigned char* mData;
  int            mWidth;
  int            mHeight;
};

class WebMapTextureLoader {
 public:
  struct Request {
    int                                  mMaxSize;
    std::string                          mStyle;
    std::optional<std::string>           mTime;
    std::optional<std::array<double, 2>> mLonRange;
    std::optional<std::array<double, 2>> mLatRange;
  };

  /// Create a new ThreadPool with the specified amount of threads.
  WebMapTextureLoader();

  ~WebMapTextureLoader();

  /// Async WMS texture loader.
  std::future<std::optional<WebMapTextureFile>> loadTextureAsync(WebMapService const& wms,
      WebMapLayer const& layer, Request const& request, std::string const& mapCache);

  /// WMS texture loader.
  std::optional<WebMapTextureFile> loadTexture(WebMapService const& wms, WebMapLayer const& layer,
      Request request, std::string const& mapCache);

  /// Load WMS texture from file using stbi.
  std::future<std::optional<WebMapTexture>> loadTextureFromFileAsync(std::string const& fileName);

 private:
  boost::filesystem::path getCachePath(
      WebMapLayer const& layer, Request const& request, std::string const& mapCache);
  std::string getRequestUrl(
      WebMapService const& wms, WebMapLayer const& layer, Request const& request);
  std::string getMimeType();

  const std::map<std::string, std::string> mMimeToExtension = {
      {"image/png", "png"}, {"image/jpeg", "jpg"}};

  std::mutex            mTextureMutex;
  cs::utils::ThreadPool mThreadPool;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_TEXTURE_LOADER_HPP
