////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TILEID_HPP
#define CSP_LOD_BODIES_TILEID_HPP

#include <boost/functional/hash/hash.hpp>
#include <functional>
#include <glm/glm.hpp>

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
  glm::int64 mPatchIdx;
  int        mLevel;
};

bool isValid(TileId const& tileId);

bool isSameLevel(TileId const& lhs, TileId const& rhs);

bool operator==(TileId const& lhs, TileId const& rhs);

std::ostream& operator<<(std::ostream& os, TileId const& tileId);
} // namespace csp::lodbodies

namespace std {

template <>
struct hash<csp::lodbodies::TileId> {
  std::size_t operator()(csp::lodbodies::TileId const& tileId) const {
    std::size_t result = 0;
    boost::hash_combine(result, tileId.level());
    boost::hash_combine(result, tileId.patchIdx());

    return result;
  }
};

} // namespace std

#endif // CSP_LOD_BODIES_TILEID_HPP
