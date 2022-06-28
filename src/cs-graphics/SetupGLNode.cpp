////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SetupGLNode.hpp"

#include <GL/glew.h>
#include <VistaMath/VistaBoundingBox.h>
#include <array>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SetupGLNode::Do() {

  // As we are using a reverse projection, we have to change the depth compare mode.
  glDepthFunc(GL_GEQUAL);

  // Also, the winding check needs to be flipped.
  glFrontFace(GL_CW);

  // In CosmoScout VR, we enable face culling per default.
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK); 

  return true;
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
