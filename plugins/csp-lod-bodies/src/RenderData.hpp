////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_RENDERDATA_HPP
#define CSP_LOD_BODIES_RENDERDATA_HPP

#include "TileBounds.hpp"
#include "TileId.hpp"
#include "TileNode.hpp"

namespace csp::lodbodies {

/// The base class for all render data of a single TileNode.
class RenderData {
 public:
  explicit RenderData(TileNode* node = nullptr);

  RenderData(RenderData const& other) = delete;
  RenderData(RenderData&& other)      = default;

  RenderData& operator=(RenderData const& other) = delete;
  RenderData& operator=(RenderData&& other) = default;

  virtual ~RenderData();

  TileNode*     getNode() const;
  void          setNode(TileNode* node);
  int           getLevel() const;
  glm::int64    getPatchIdx() const;
  TileId const& getTileId() const;

  int  getTexLayer() const;
  void setTexLayer(int layer);

  int  getLastFrame() const;
  void setLastFrame(int frame);
  int  getAge(int frame) const;

  BoundingBox<double> const& getBounds() const;
  void                       setBounds(BoundingBox<double> const& tb);
  void                       removeBounds();
  bool                       hasBounds() const;

 protected:
  BoundingBox<double> mTb;
  bool                mHasBounds{};

 private:
  TileNode* mNode{};
  int       mTexLayer{};
  int       mLastFrame{};
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_RENDERDATA_HPP
