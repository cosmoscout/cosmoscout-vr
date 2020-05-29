////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_UPDATEBOUNDSVISITOR_HPP
#define CSP_LOD_BODIES_UPDATEBOUNDSVISITOR_HPP

#include "TileVisitor.hpp"

namespace csp::lodbodies {

struct PlanetParameters;
class TreeManagerBase;

/// DocTODO isn't used anywhere in the project.
class UpdateBoundsVisitor : public TileVisitor<UpdateBoundsVisitor> {
 public:
  explicit UpdateBoundsVisitor(TreeManagerBase* treeMgrDEM, PlanetParameters const& params);

 protected:
  bool preTraverse() override;
  bool preVisitRoot(TileId const& tileId) override;
  bool preVisit(TileId const& tileId) override;

  friend class TileVisitor<UpdateBoundsVisitor>;

  TreeManagerBase*        mTreeMgrDEM;
  PlanetParameters const* mParams;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_UPDATEBOUNDSVISITOR_HPP
