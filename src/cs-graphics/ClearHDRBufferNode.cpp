////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ClearHDRBufferNode.hpp"

#include "HDRBuffer.hpp"

#include <VistaMath/VistaBoundingBox.h>
#include <limits>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

ClearHDRBufferNode::ClearHDRBufferNode(std::shared_ptr<HDRBuffer> const& hdrBuffer)
    : mHDRBuffer(hdrBuffer) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ClearHDRBufferNode::ClearHDRBufferNode::Do() {
  mHDRBuffer->clear();
  mHDRBuffer->bind();
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ClearHDRBufferNode::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float min(std::numeric_limits<float>::min());
  float max(std::numeric_limits<float>::max());
  float fMin[3] = {min, min, min};
  float fMax[3] = {max, max, max};

  oBoundingBox.SetBounds(fMin, fMax);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
