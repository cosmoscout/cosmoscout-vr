////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILEID_HPP
#define CSP_LOD_BODIES_TILEID_HPP

#include <functional>
#include <glm/glm.hpp>
#include <ostream>

namespace csp::lodbodies {

/// Identifier for a tile, consisting of a level and a patch index within that level.
class TileId {
 public:
  explicit TileId();
  explicit TileId(int level, glm::int64 patchIdx);

  void reset();

  /// The level of subdivision of the tile.
  int  level() const;
  void level(int level);

  glm::int64 patchIdx() const;
  void       patchIdx(glm::int64 pi);

 private:
  glm::int64 mPatchIdx{-1};
  int        mLevel{-1};
};

bool isValid(TileId const& tileId);

bool isSameLevel(TileId const& lhs, TileId const& rhs);

bool operator==(TileId const& lhs, TileId const& rhs);

std::ostream& operator<<(std::ostream& os, TileId const& tileId);
} // namespace csp::lodbodies

template <typename T>
void hash_combine(std::size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {

template <>
struct hash<csp::lodbodies::TileId> {
  std::size_t operator()(csp::lodbodies::TileId const& tileId) const {
    std::size_t result = 0;
    hash_combine(result, tileId.level());
    hash_combine(result, tileId.patchIdx());

    return result;
  }
};

} // namespace std

#endif // CSP_LOD_BODIES_TILEID_HPP
