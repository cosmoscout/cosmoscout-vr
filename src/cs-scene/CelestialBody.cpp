////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CelestialBody.hpp"

namespace cs::scene {

////////////////////////////////////////////////////////////////////////////////////////////////////

CelestialBody::CelestialBody(std::string const& sCenterName, std::string const& sFrameName,
    glm::dvec3 const& radii, double tStartExistence, double tEndExistence)
    : CelestialObject(sCenterName, sFrameName, radii, tStartExistence, tEndExistence) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::scene
