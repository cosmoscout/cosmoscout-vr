////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_RENDERDATAIMG_HPP
#define CSP_LOD_BODIES_RENDERDATAIMG_HPP

#include "RenderData.hpp"

namespace csp::lodbodies {

/// Render data for image data.
class RenderDataImg : public RenderData {
 public:
  explicit RenderDataImg(TileNode* node = nullptr);
  ~RenderDataImg() override;

  RenderDataImg(RenderDataImg const& other) = delete;
  RenderDataImg(RenderDataImg&& other)      = delete;

  RenderDataImg& operator=(RenderDataImg const& other) = delete;
  RenderDataImg& operator=(RenderDataImg&& other) = delete;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_RENDERDATAIMG_HPP
