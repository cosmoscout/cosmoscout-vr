////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_SIMPLE_OBJECTS_UTILS_HPP
#define CSP_SIMPLE_OBJECTS_UTILS_HPP

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include "../../../src/cs-scene/CelestialBody.hpp"

namespace csp::simpleobjects { 

namespace utils { 
    
    // Returns the normalized normal of a sloped surface. Is usually used with lod bodies.
    // Offset in meters
    glm::dvec3 getSurfaceNormal(const glm::dvec2 &lngLat, std::shared_ptr<cs::scene::CelestialBody> &body, const double offset = 0.9F); 

    // wrapper around cs::utils::convert::toCartesian for easier 
    // access to position from getSurfaceNormal() 
    glm::dvec3 getLngLatPositionOnBody(const glm::dvec2 &lngLat, std::shared_ptr<cs::scene::CelestialBody> &body);

    glm::dquat normalToRotation(const glm::dvec3 &normal);

} //namespace utils

} // namespace csp::simplobjects

#endif // CSP_SIMPLE_OBJECTS_UTILS_HPP