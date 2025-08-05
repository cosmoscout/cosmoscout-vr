////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_BASE_TILE_DATA_HPP
#define CSP_LOD_BODIES_BASE_TILE_DATA_HPP

#include "BoundingBox.hpp"
#include "MinMaxPyramid.hpp"
#include "TileDataType.hpp"
#include "TileId.hpp"

#include <memory>
#include <typeinfo>

namespace csp::lodbodies {

/// Abstract base class for data tiles. This is used to access data for single tile. Actual data is
/// held by classes derived from this one.
class BaseTileData {
 public:
  virtual ~BaseTileData() = default;

  BaseTileData(BaseTileData const& other) = delete;
  BaseTileData(BaseTileData&& other)      = default;

  BaseTileData& operator=(BaseTileData const& other) = delete;
  BaseTileData& operator=(BaseTileData&& other)      = default;

  /// Returns the enum value for the data type stored in this tile.
  virtual TileDataType getDataType() const = 0;

  /// Returns read only pointer to data stored in this tile as T*.
  template <typename T>
  T const* getTypedPtr() const;

  /// Returns pointer to data stored in this tile as T*.
  template <typename T>
  T* getTypedPtr();

  /// Returns read only pointer to data stored in this tile.
  virtual void const* getDataPtr() const = 0;

  /// Returns pointer to data stored in this tile.
  virtual void* getDataPtr() = 0;

  /// Returns the resolution given to the tile at construction time.
  uint32_t getResolution() const;

  /// Once uploaded to the GPU, this will return the layer in the 3D texture array this data is
  /// stored in.
  int  getTexLayer() const;
  void setTexLayer(int layer);

 protected:
  explicit BaseTileData(uint32_t resolution);

 private:
  uint32_t mResolution;
  int      mTexLayer{-1};
};

template <typename T>
T const* BaseTileData::getTypedPtr() const {
  return static_cast<T const*>(getDataPtr());
}

template <typename T>
T* BaseTileData::getTypedPtr() {
  return static_cast<T*>(getDataPtr());
}

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_BASE_TILE_DATA_HPP
