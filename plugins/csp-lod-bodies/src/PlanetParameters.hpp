////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_PLANETPARAMETERS_HPP
#define CSP_LOD_BODIES_PLANETPARAMETERS_HPP

#include <glm/glm.hpp>

namespace csp::lodbodies {

/// Holds values describing a planets parameters.
struct PlanetParameters {
  glm::dvec3 mRadii       = glm::dvec3(1.0);
  double     mHeightScale = 1.0;  ///< The level of exaggeration of the surface height.
  double     mLodFactor   = 50.0; ///< DocTODO

  int mMinLevel = 0; ///< The minimum LOD level.
  int mMaxLevel = 0; ///< The maximum LOD level.
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_PLANETPARAMETERS_HPP
