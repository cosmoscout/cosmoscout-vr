////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_RENDERDATADEM_HPP
#define CSP_LOD_BODIES_RENDERDATADEM_HPP

#include "RenderData.hpp"
#include "TileBounds.hpp"

namespace csp::lodbodies {

/// Render data for elevation. It also holds additional information about the edges to prevent
/// elevation jumps due to different resolutions.
class RenderDataDEM : public RenderData {
 public:
  enum Flags { eRender = 0x01 };

  explicit RenderDataDEM(TileNode* node = nullptr);

  RenderDataDEM(RenderDataDEM const& other) = delete;
  RenderDataDEM(RenderDataDEM&& other)      = delete;

  RenderDataDEM& operator=(RenderDataDEM const& other) = delete;
  RenderDataDEM& operator=(RenderDataDEM&& other) = delete;

  ~RenderDataDEM() override;

  int  getEdgeDelta(int idx) const;
  void setEdgeDelta(int idx, int delta);
  void resetEdgeDeltas();

  RenderDataDEM* getEdgeRData(int idx) const;
  void           setEdgeRData(int idx, RenderDataDEM* rdata);
  void           resetEdgeRData();

  void addFlag(Flags flag);
  void subFlag(Flags flag);
  bool testFlag(Flags flag) const;

  glm::uint8 getFlags() const;
  void       clearFlags();

 private:
  std::array<glm::int8, 4>      mLodDeltas;
  std::array<RenderDataDEM*, 4> mEdgeRData;
  glm::uint8                    mFlags;
};

RenderDataDEM::Flags operator&(RenderDataDEM::Flags lhs, RenderDataDEM::Flags rhs);
RenderDataDEM::Flags operator|(RenderDataDEM::Flags lhs, RenderDataDEM::Flags rhs);

RenderDataDEM::Flags& operator&=(RenderDataDEM::Flags& lhs, RenderDataDEM::Flags rhs);
RenderDataDEM::Flags& operator|=(RenderDataDEM::Flags& lhs, RenderDataDEM::Flags rhs);

RenderDataDEM::Flags operator~(RenderDataDEM::Flags lhs);

std::ostream& operator<<(std::ostream& os, RenderDataDEM::Flags flags);

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_RENDERDATADEM_HPP
