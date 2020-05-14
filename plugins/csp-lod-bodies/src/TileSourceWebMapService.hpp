////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TILESOURCEWMS_HPP
#define CSP_LOD_BODIES_TILESOURCEWMS_HPP

#include "../../../src/cs-utils/ThreadPool.hpp"
#include "Tile.hpp"
#include "TileSource.hpp"

#include <cstdio>
#include <string>

namespace csp::lodbodies {

/// The data of the tiles is fetched via a web map service.
class TileSourceWebMapService : public TileSource {
 public:
  TileSourceWebMapService();

  TileSourceWebMapService(TileSourceWebMapService const& other) = delete;
  TileSourceWebMapService(TileSourceWebMapService&& other)      = delete;

  TileSourceWebMapService& operator=(TileSourceWebMapService const& other) = delete;
  TileSourceWebMapService& operator=(TileSourceWebMapService&& other) = delete;

  ~TileSourceWebMapService() override = default;

  void init() override {
  }

  void fini() override {
  }

  TileNode* loadTile(int level, glm::int64 patchIdx) override;

  void loadTileAsync(int level, glm::int64 patchIdx, OnLoadCallback cb) override;
  int  getPendingRequests() override;

  void     setMaxLevel(uint32_t maxLevel);
  uint32_t getMaxLevel() const;

  void               setCacheDirectory(std::string const& cacheDirectory);
  std::string const& getCacheDirectory() const;

  void               setLayers(std::string const& layers);
  std::string const& getLayers() const;

  void               setUrl(std::string const& url);
  std::string const& getUrl() const;

  void         setDataType(TileDataType type);
  TileDataType getDataType() const override;

  bool isSame(TileSource const* other) const override;

  /// These can be used to pre-populate the local cache, returns true if the tile is on the diagonal
  /// of base patch 4 (the one which is cut in two halves).
  static bool getXY(int level, glm::int64 patchIdx, int& x, int& y);
  std::string loadData(int level, int x, int y);

 private:
  static std::mutex mTileSystemMutex;

  cs::utils::ThreadPool mThreadPool;
  std::string           mUrl;
  std::string           mCache = "cache/img";
  std::string           mLayers;
  TileDataType          mFormat   = TileDataType::eU8Vec3;
  uint32_t              mMaxLevel = 10;
};
} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILESOURCEWMS_HPP
