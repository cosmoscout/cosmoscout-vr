////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_PLANETPARAMETERS_HPP
#define CSP_LOD_BODIES_PLANETPARAMETERS_HPP

namespace csp::lodbodies {

/// Holds values describing a planets parameters.
struct PlanetParameters {
  double mPolarRadius      = 1.0;
  double mEquatorialRadius = 1.0;

  double mHeightScale = 1.0;  ///< The level of exaggeration of the surface height.
  double mLodFactor   = 50.0; ///< DocTODO

  int mMinLevel = 0; ///< The minimum LOD level.
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_PLANETPARAMETERS_HPP
