////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "utils.hpp"
#include "logger.hpp"

#include "../../../src/cs-scene/CelestialBody.hpp"
#include "../../../src/cs-utils/convert.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

namespace csp::simpleobjects::utils {

    using glm::dvec3, glm::dvec2;


    dvec3 getSurfaceNormal(const dvec2 &lngLat, std::shared_ptr<cs::scene::CelestialBody> &body, const double offset) { 
        
        dvec3 radii = body->getRadii();
        double offsetAngle;
        
        if( radii[0] == radii[1] && radii[0] == radii[2] ) {
            offsetAngle = offset / radii[0];
        } else {
            offsetAngle = offset / sqrt( (radii[0] * radii[0] + radii[1] * radii[1] + radii[2] * radii[2]) / 3.0F );
        }

        dvec2 offLL1(lngLat.x - offsetAngle, lngLat.y);
        dvec2 offLL2(lngLat.x + offsetAngle, lngLat.y);
        dvec2 offLL3(lngLat.x, lngLat.y - offsetAngle);
        dvec2 offLL4(lngLat.x, lngLat.y + offsetAngle);

        dvec3 surfaceVec1 = getLngLatPositionOnBody(offLL2, body) - getLngLatPositionOnBody(offLL1, body);
        dvec3 surfaceVec2 = getLngLatPositionOnBody(offLL4, body) - getLngLatPositionOnBody(offLL3, body);

        return glm::normalize(glm::cross(surfaceVec1, surfaceVec2));
    }

    
    dvec3 getLngLatPositionOnBody(const dvec2 &lngLat, std::shared_ptr<cs::scene::CelestialBody> &body) {         
        return cs::utils::convert::toCartesian(lngLat, body->getRadii(), body->getHeight(lngLat));
    }


    glm::dquat normalToRotation(const dvec3 &normal) { 
        
        dvec3 north = dvec3(0,1,0);
        dvec3 z = glm::normalize(glm::cross(north, normal));
        north        = glm::normalize(glm::cross(normal, z));
            
        return glm::toQuat(glm::dmat3(north, normal, z));
    }


} // namespace cs::simpleobjects::utils