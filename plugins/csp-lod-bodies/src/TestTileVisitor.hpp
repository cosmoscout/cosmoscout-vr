////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TESTTILEVISITOR_HPP
#define CSP_LOD_BODIES_TESTTILEVISITOR_HPP

#include "TileId.hpp"
#include "TileVisitor.hpp"

#include <vector>

namespace csp::lodbodies {

/// DocTODO it isn't used anywhere...
class TestTileVisitor : public TileVisitor<TestTileVisitor> {
 public:
  explicit TestTileVisitor(TileQuadTree* treeDEM, TileQuadTree* treeIMG = nullptr);

  std::vector<TileId> const& getLoadTilesDEM() const;
  std::vector<TileId> const& getLoadTilesIMG() const;

 private:
  bool preTraverse() override;

  bool preVisitRoot(TileId const& tileId) override;
  bool preVisit(TileId const& tileId) override;
  void postVisit(TileId const& tileId) override;

  bool visitLevel(TileId const& tileId);
  bool refineTile();

  std::vector<TileId> mLoadTilesDEM;
  std::vector<TileId> mLoadTilesIMG;

  friend class TileVisitor<TestTileVisitor>;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TESTTILEVISITOR_HPP
