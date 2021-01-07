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
};

class WebMapTextureLoader {
 public:
  /// Create a new ThreadPool with the specified amount of threads.
  WebMapTextureLoader();

  ~WebMapTextureLoader();

  /// Async WMS texture loader.
  std::future<std::string> loadTextureAsync(WebMapService const& wms, WebMapLayer const& layer,
      std::string const& time, std::string const& mapCache, int const& maxSize,
      std::array<double, 2> lonRange, std::array<double, 2> latRange);
  std::future<std::string> loadTextureAsync(WebMapService const& wms, WebMapLayer const& layer,
      std::string const& time, std::string const& mapCache, int const& maxSize);

  /// WMS texture loader.
  std::string loadTexture(WebMapService const& wms, WebMapLayer const& layer,
      std::string const& time, std::string const& mapCache, int const& maxSize,
      std::array<double, 2> lonRange, std::array<double, 2> latRange);
  std::string loadTexture(WebMapService const& wms, WebMapLayer const& layer,
      std::string const& time, std::string const& mapCache, int const& maxSize);

  /// Load WMS texture from file using stbi.
  std::future<WebMapTexture> loadTextureFromFileAsync(std::string const& fileName);

 private:
  boost::filesystem::path getCachePath(WebMapLayer const& layer, std::string const& time,
      std::string const& mapCache, std::array<double, 2> const& lonRange,
      std::array<double, 2> const& latRange, std::string const& mime);
  std::string             getRequestUrl(WebMapService const& wms, WebMapLayer const& layer,
                  std::string const& time, int const& maxSize, std::array<double, 2> const& lonRange,
                  std::array<double, 2> const& latRange, std::string const& mime);
  std::string             getMimeType();

  const std::map<std::string, std::string> mMimeToExtension = {
      {"image/png", "png"}, {"image/jpeg", "jpg"}};

  std::mutex            mTextureMutex;
  cs::utils::ThreadPool mThreadPool;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_TEXTURE_LOADER_HPP
