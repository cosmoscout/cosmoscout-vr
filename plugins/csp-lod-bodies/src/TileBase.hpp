////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TILEBASE_HPP
#define CSP_LOD_BODIES_TILEBASE_HPP

#include "MinMaxPyramid.hpp"
#include "TileDataType.hpp"
#include "TileId.hpp"

#include <boost/noncopyable.hpp>
#include <memory>
#include <typeinfo>

namespace csp::lodbodies {

/// Abstract base class for data tiles in the HEALPix scheme. A tile stores data samples for a
/// HEALPix patch at a given subdivision level. Actual data is held by classes derived from this
/// one.
class TileBase : private boost::noncopyable {
 public:
  /// Number of samples belonging to this tiles, x direction.
  static int const sOwnSizeX = 256;

  /// Number of samples belonging to this tiles, y direction.
  static int const sOwnSizeY = 256;

  /// Number of samples stored in this tiles (i.e. including border samples), x direction.
  static int const SizeX = sOwnSizeX + 1;

  /// Number of samples stored in this tiles (i.e. including border samples), y direction.
  static int const SizeY = sOwnSizeY + 1;

  virtual ~TileBase() = default;

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

  /// The level of subdivision of the tile.
  int  getLevel() const;
  void setLevel(int level);

  glm::int64 getPatchIdx() const;
  void       setPatchIdx(glm::int64 patchIdx);

  MinMaxPyramid* getMinMaxPyramid() const;
  void           setMinMaxPyramid(std::unique_ptr<MinMaxPyramid> pyramid);

 protected:
  explicit TileBase(int level, glm::int64 patchIdx);

  TileId mTileId;

 private:
  std::unique_ptr<MinMaxPyramid> mMinMaxPyramid;
};

template <typename T>
T const* TileBase::getTypedPtr() const {
  return static_cast<T const*>(getDataPtr());
}

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILEBASE_HPP
