////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_BODIES_RENDER_DATA_HPP
#define CSP_CESIUM_BODIES_RENDER_DATA_HPP

#include <cstdint>
#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>

namespace csp::cesiumbodies {

struct TextureData {
  std::vector<std::byte> pixels;
  int32_t                width       = 0;
  int32_t                height      = 0;
  int32_t                channels    = 4;
  int32_t                sourceIndex = -1;
  int32_t                wrapS       = GL_REPEAT;
  int32_t                wrapT       = GL_REPEAT;
  int32_t                minFilter   = GL_LINEAR_MIPMAP_LINEAR;
  int32_t                magFilter   = GL_LINEAR;
  GLuint                 textureId   = 0;
};

struct DrawBatch {
  uint32_t firstIndex  = 0;
  uint32_t indexCount  = 0;
  int32_t  textureSlot = -1;
};

/// This struct carries extracted mesh data from the CPU worker thread to the main (GPU) thread.
/// It lives on the heap and is passed as void*.
struct CesiumRenderData {
  std::vector<float>    vertices; ///< Interleaved: [Px,Py,Pz, U,V, R,G,B,A] = 9 floats
  std::vector<uint32_t> indices;  ///< Triangle indices (always uint32_t)

  std::vector<TextureData> textures;
  std::vector<DrawBatch>   batches;

  GLuint vao = 0;
  GLuint vbo = 0;
  GLuint ebo = 0;

  /// Corrected tile-to-ECEF transform (with RTC center + up-axis applied).
  /// The renderer uses this instead of raw pTile->getTransform().
  glm::dmat4 tileTransform{1.0};

  /// CPU-side copies retained for getHeight() / getIntersection() queries.
  /// Only positions + indices are kept — normals, UVs, colors are discarded.
  std::vector<glm::dvec3> cpuPositions;
  std::vector<uint32_t>   cpuIndices;
};

} // namespace csp::cesiumbodies

#endif // CSP_CESIUM_BODIES_RENDER_DATA_HPP
