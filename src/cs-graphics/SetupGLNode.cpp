////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SetupGLNode.hpp"

#include <VistaMath/VistaBoundingBox.h>
#include <array>
#include <GL/glew.h>

namespace cs::graphics {
    
////////////////////////////////////////////////////////////////////////////////////////////////////

bool SetupGLNode::Do() {
    glDepthFunc(GL_GEQUAL);
    glClearDepth(0.0f);
     // glEnable(GL_CULL_FACE);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

  bool SetupGLNode::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
      float min(std::numeric_limits<float>::lowest());
  float max(std::numeric_limits<float>::max());

  std::array fMin{min, min, min};
  std::array fMax{max, max, max};

  oBoundingBox.SetBounds(fMin.data(), fMax.data());

  return true;
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics

