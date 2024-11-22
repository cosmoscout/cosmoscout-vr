////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VIRTUAL_SATELLITE_RENDER_TYPES_HPP
#define CSP_VIRTUAL_SATELLITE_RENDER_TYPES_HPP

namespace csp::virtualsatellite {

struct Box {
  glm::vec3 pos;
  glm::quat rot;
  glm::vec3 size;
  glm::vec4 color;
};

struct Sphere {
  glm::vec3 pos;
  float     radius;
  glm::vec4 color;
};

} // namespace csp::virtualsatellite

#endif // CSP_VIRTUAL_SATELLITE_RENDER_TYPES_HPP
