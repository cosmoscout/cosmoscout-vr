////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TREEMANAGER_HPP
#define CSP_LOD_BODIES_TREEMANAGER_HPP

#include "RenderDataDEM.hpp"
#include "RenderDataImg.hpp"
#include "TreeManagerBase.hpp"

#include <boost/cast.hpp>
#include <boost/pool/object_pool.hpp>

namespace csp::lodbodies {

/// Implements management of a TileQuadTree with associated data of type RDataT (which must be
/// derived from RenderData). Almost all functionality is implemented in the base class
/// TreeManagerBase, only allocation and release of the associated data for a node is managed here.
template <typename RDataT>
class TreeManager : public TreeManagerBase {
 public:
  explicit TreeManager(
      PlanetParameters const& params, std::shared_ptr<GLResources> const& glResources);

  TreeManager(TreeManager const& other) = delete;
  TreeManager(TreeManager&& other)      = delete;

  TreeManager& operator=(TreeManager const& other) = delete;
  TreeManager& operator=(TreeManager&& other) = delete;

  ~TreeManager() override;

 protected:
  RenderData* allocateRenderData(TileNode* node) override;
  void        releaseRenderData(RenderData* rdata) override;

  boost::object_pool<RDataT> mPool;
};

template <>
/* virtual */ RenderData* TreeManager<RenderDataDEM>::allocateRenderData(TileNode* node);

template <typename RDataT>
/* explicit */
TreeManager<RDataT>::TreeManager(
    PlanetParameters const& params, std::shared_ptr<GLResources> const& glResources)
    : TreeManagerBase(params, glResources)
    , mPool() {
}

template <typename RDataT>
/* virtual */
TreeManager<RDataT>::~TreeManager() = default;

template <typename RDataT>
/* virtual */ RenderData* TreeManager<RDataT>::allocateRenderData(TileNode* node) {
  RDataT* rdata = mPool.construct();

  rdata->setNode(node);
  rdata->setLastFrame(0);

  return rdata;
}

template <typename RDataT>
/* virtual */ void TreeManager<RDataT>::releaseRenderData(RenderData* rdata) {
  RDataT* rd = boost::polymorphic_downcast<RDataT*>(rdata);

  mPool.destroy(rd);
}

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TREEMANAGER_HPP
