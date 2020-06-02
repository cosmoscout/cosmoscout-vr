////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_HEALPIX_HPP
#define CSP_LOD_BODIES_HEALPIX_HPP

#include "TileId.hpp"

/// @file
/// Implementation of the HEALPix sphere tesselation scheme. Based on the paper: "HEALPix: A
/// Framework for High-Resolution Discretization and fast Analysis of Data Distributed on the
/// Sphere" by K.M. Gorski et al.

namespace csp::lodbodies {

class HEALPix;

/// Canonical order of patch edges is defined by this enum. Whenever edges (or data associated with
/// edges) are stored by the HEALPix or HEALPixLevel types this order of edges is used.
enum class EdgeDirection {
  eNorthEast = 0x00,
  eNorthWest = 0x01,
  eSouthWest = 0x02,
  eSouthEast = 0x03
};

std::ostream& operator<<(std::ostream& os, EdgeDirection ed);

/// Describes one subdivision level in the HEALPix scheme and allows calculation of patch locations,
/// boundary points and transformation between different numbering schemes. It is not necessary
/// (or possible) to create instances of this class directly, use HEALPix to access an instance for
/// a desired level.
class HEALPixLevel {
 public:
  int getLevel() const;

  /// Returns the number of sub patches along the edge of a base patch at this sub division level.
  glm::int64 getNSide() const;

  /// Returns number of patches in a base patch at this sub division level. This is simply
  /// getNSide() ^ 2.
  glm::int64 getPatchCount() const;

  /// Returns number of patches in all base patches at this sub division level. This is simply
  /// 12 * getPatchCount().
  glm::int64 getTotalPatchCount() const;

  /// Returns the base patch in which the patch with given patchIdx is contained in.
  int getBasePatch(glm::int64 patchIdx) const;

  /// Returns coordinates of patch patchIdx as base patch index and integral (x, y) indices
  /// inside the base patch, where (x, y) are in [0, getNSide() - 1].
  glm::i64vec3 getBaseXY(glm::int64 patchIdx) const;

  /// Returns the patch index that belongs to the patch given by bxy where bxy[0] is the base patch
  /// index and bxy[1], bxy[2] are the integral (x, y) indices inside the base patch.
  glm::int64 getPatchIdx(glm::i64vec3 const& bxy) const;

  /// Returns center of patch patchIdx in geodetic coordinates (lng, lat) in radians.
  glm::dvec2 getCenterLngLat(glm::int64 patchIdx) const;

  /// Returns center of patch patchIdx in cartesian coordinates (x, y, z).
  glm::dvec3 getCenterCartesian(glm::int64 patchIdx, double radiusE, double radiusP) const;

  /// Returns corners of patch patchIdx in the order N, W, S, E in geodetic coordinates (lng, lat)
  /// in radians.
  std::array<glm::dvec2, 4> getCornersLngLat(glm::int64 patchIdx) const;

  /// Returns corners of patch patchIdx in the order N, W, S, E in cartesian coordinates (x, y, z).
  ///
  /// The vectors have unit length.
  std::array<glm::dvec3, 4> getCornersCartesian(
      glm::int64 patchIdx, double radiusE, double radiusP) const;

  /// Returns the center points of the edges of patch patchIdx in the order NE, NW, SW, SE in
  /// geodetic coordinates (lng, lat) in radians.
  std::array<glm::dvec2, 4> getEdgeCentersLngLat(glm::int64 patchIdx) const;

  /// Returns the center points of the edges of patch patchIdx in the order NE, NW, SW, SE in
  /// cartesian coordinates (x, y, z).
  std::array<glm::dvec3, 4> getEdgeCentersCartesian(
      glm::int64 patchIdx, double radiusE, double radiusP) const;

  /// Returns the patch indices of the NE (+X), NW (+Y), SW (-X), SE (-Y) neighbours of patch
  /// patchIdx.
  std::array<glm::int64, 4> getNeighbours(glm::int64 patchIdx) const;

  int        getF1(glm::int64 patchIdx) const;
  int        getF2(glm::int64 patchIdx) const;
  glm::dvec3 getPatchOffsetScale(glm::int64 patchIdx) const;

 private:
  explicit HEALPixLevel(int level);

  /// Returns the value formed by the even numbered bits of value. That is if value has binary
  /// digits "...ihgfedcba" returns the value "...igeca".
  static glm::int64 extractEvenBits(glm::int64 value);

  /// Returns the value formed by the odd numbered bits of value.
  static glm::int64 extractOddBits(glm::int64 value);

  static void extractBits(glm::int64 value, glm::int64& evenBits, glm::int64& oddBits);

  static glm::int64 replaceEvenBits(glm::int64 value);
  static glm::int64 replaceOddBits(glm::int64 value);
  static glm::int64 replaceBits(glm::int64 evenBits, glm::int64 oddBits);

  glm::dvec2 bxy2geo(glm::i64vec3 const& bxy) const;

  /// Look-up-table for extraction of the odd/even bits from an integer.
  static std::array<glm::uint16, 256> const sCompressLUT;

  /// Look-up-table for setting of the odd/even bits of a 64bit integer from a 32bit integer.
  static std::array<glm::uint64, 256> const sExpandLUT;

  /// Look-up-table for offsets of neighbour patches.
  static std::array<glm::i64vec2, 4> const sNeighbourOffsetLUT;

  /// Look-up-table for neighbours of the 12 base patches.
  static std::array<glm::ivec4, 12> const sBaseNeighbourLUT;

  /// Look-up table for f1, f2 parameters.
  static std::array<glm::uint16, 12> const sF1LUT;
  static std::array<glm::uint16, 12> const sF2LUT;

  /// Subdivision level. Level 0 is the root (i.e. the 12 base patches), on each successive level a
  /// patch is split into four sub patches.
  int mLevel;

  /// Number of sub patches along one edge of a base patch. A base patch is split into
  /// mNSide * mNSide sub patches at this level. This number is a power of two,
  /// specifically: mNSide = 2^level_
  glm::int64 mNSide;

  friend class HEALPix;
};

class HEALPix {
 public:
  static HEALPixLevel const& getLevel(int level);

  static glm::int64   getNSide(TileId const& tileId);
  static int          getBasePatch(TileId const& tileId);
  static glm::i64vec3 getBaseXY(TileId const& tileId);

  static glm::dvec2 getCenterLngLat(TileId const& tileId);
  static glm::dvec3 getCenterCartesian(TileId const& tileId, double radiusE, double radiusP);

  static std::array<glm::dvec2, 4> getCornersLngLat(TileId const& tileId);
  static std::array<glm::dvec3, 4> getCornersCartesian(
      TileId const& tileId, double radiusE, double radiusP);

  static std::array<glm::dvec2, 4> getEdgeCentersLngLat(TileId const& tileId);
  static std::array<glm::dvec3, 4> getEdgeCentersCartesian(
      TileId const& tileId, double radiusE, double radiusP);

  static std::array<glm::int64, 4> getNeighbours(TileId const& tileId);
  static std::array<TileId, 4>     getNeighbourIds(TileId const& tileId);

  static int        getF1(TileId const& tileId);
  static int        getF2(TileId const& tileId);
  static glm::dvec3 getPatchOffsetScale(TileId const& tileId);

  /// Converts the given coordinates basePatchIdx, x, y to geodetic coordinates (lng, lat) in
  /// radians. x, y are in [0, 1].
  static glm::dvec2 convertBaseXY2LngLat(int basePatchIdx, double x, double y);
  static glm::dvec2 convertBaseLngLat2XY(int basePatchIdx, glm::dvec2 const& lngLat);
  static int        convertLngLat2Base(glm::dvec2 const& lngLat);

  /// Returns the index of the root patch for the tile with given tileId.
  static int getRootIdx(TileId const& tileId);

  /// Returns the index of the child at level that corresponds to tileId.
  static int getChildIdxAtLevel(TileId const& tileId, int level);

  /// Returns the index of the child at the lowest level that corresponds to tileId.
  static int getChildIdx(TileId const& tileId);

  /// Returns the level of the children for a parent node with given parentId.
  static int getChildLevel(TileId const& parentId);

  /// Returns the patch index of child childIdx for a parent node with given parentId.
  static glm::int64 getChildPatchIdx(TileId const& parentId, int childIdx);

  /// Returns the TileId of child childIdx for a parent node with given parentId.
  static TileId getChildTileId(TileId const& parentId, int childIdx);

  /// Returns the level of the parent node for a node with given childId.
  static int getParentLevel(TileId const& childId);

  /// Returns the patch index of the parent node for a node with given childId.
  static glm::int64 getParentPatchIdx(TileId const& childId);

  /// Returns the TileId of the parent node for a node with given childId.
  static TileId getParentTileId(TileId const& childId);

 private:
  static std::array<HEALPixLevel, 20> const sLevels;
};
} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_HEALPIX_HPP
