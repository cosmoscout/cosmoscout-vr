////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILEBASE_HPP
#define CSP_LOD_BODIES_TILEBASE_HPP

#include "MinMaxPyramid.hpp"
#include "TileDataType.hpp"
#include "TileId.hpp"

#include <memory>
#include <typeinfo>

namespace csp::lodbodies {

/// Abstract base class for data tiles in the HEALPix scheme. A tile stores data samples for a
/// HEALPix patch at a given subdivision level. Actual data is held by classes derived from this
/// one.
class TileBase {
 public:
  virtual ~TileBase() = default;

  TileBase(TileBase const& other) = delete;
  TileBase(TileBase&& other)      = default;

  TileBase& operator=(TileBase const& other) = delete;
  TileBase& operator=(TileBase&& other) = default;

  /// Returns std::type_info for the data type stored in this tile.
  virtual std::type_info const& getTypeId() const = 0;

  /// Returns the enum value for the data type stored in this tile.
  virtual TileDataType getDataType() const = 0;

  /// Returns read only pointer to data stored in this tile as T*.
  template <typename T>
  T const* getTypedPtr() const;

  /// Returns read only pointer to data stored in this tile.
  virtual void const* getDataPtr() const = 0;

  TileId const& getTileId() const;
  void          setTileId(TileId const& tileId);

  uint32_t getResolution() const;

  /// The level of subdivision of the tile.
  int  getLevel() const;
  void setLevel(int level);

  glm::int64 getPatchIdx() const;
  void       setPatchIdx(glm::int64 patchIdx);

  MinMaxPyramid* getMinMaxPyramid() const;
  void           setMinMaxPyramid(std::unique_ptr<MinMaxPyramid> pyramid);

 protected:
  explicit TileBase(int level, glm::int64 patchIdx, uint32_t resolution);

  TileId mTileId;

 private:
  std::unique_ptr<MinMaxPyramid> mMinMaxPyramid;
  uint32_t                       mResolution;
};

template <typename T>
T const* TileBase::getTypedPtr() const {
  return static_cast<T const*>(getDataPtr());
}

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILEBASE_HPP
