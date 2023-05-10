////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILESOURCEWMS_HPP
#define CSP_LOD_BODIES_TILESOURCEWMS_HPP

#include "../../../src/cs-utils/ThreadPool.hpp"
#include "TileData.hpp"
#include "TileSource.hpp"

#include <cstdio>
#include <optional>
#include <string>

namespace csp::lodbodies {

/// The data of the tiles is fetched via a web map service.
class TileSourceWebMapService : public TileSource {
 public:
  TileSourceWebMapService(uint32_t resolution);

  TileSourceWebMapService(TileSourceWebMapService const& other) = delete;
  TileSourceWebMapService(TileSourceWebMapService&& other)      = delete;

  TileSourceWebMapService& operator=(TileSourceWebMapService const& other) = delete;
  TileSourceWebMapService& operator=(TileSourceWebMapService&& other) = delete;

  ~TileSourceWebMapService() override = default;

  void init() override {
  }

  void fini() override {
  }

  std::shared_ptr<BaseTileData> loadTile(TileId const& tileId) override;

  void loadTileAsync(TileId const& tileId, OnLoadCallback cb) override;
  int  getPendingRequests() override;

  uint32_t getResolution() const;

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
  static bool getXY(TileId const& tileId, int& x, int& y);

  // This downloads the tile with the given coordinates from the MapServer. It is stored in the
  // local map cache and the resulting file name is returned. If the tile is already present in the
  // map cache, no request is made and the cahce file name is returned immediately. It may happen
  // that a tile cannot be downloaded (e.g. if the server is offline) - in this case no error is
  // thrown but std::nullopt is returned. In several other cases (e.g. cache directory is not
  // writable) a std::runtime_error is thrown.
  std::optional<std::string> loadData(TileId const& tileId, int x, int y);

 private:
  static std::mutex mFileSystemMutex;

  cs::utils::ThreadPool mThreadPool;
  std::string           mUrl;
  std::string           mCache = "cache/img";
  std::string           mLayers;
  TileDataType          mFormat = TileDataType::eColor;
  uint32_t              mResolution;
};
} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILESOURCEWMS_HPP
