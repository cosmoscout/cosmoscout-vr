////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WFS_OVERLAYS_WFS_TYPES_HPP
#define CSP_WFS_OVERLAYS_WFS_TYPES_HPP

#include <vector>
#include <string>
#include <array>

namespace csp::wfsoverlays {
    struct Geometry {
    std::string type;
    std::array<double, 2> coordinates;
    }; 

    struct Feature {  
    std::string type;
    std::string id;
    Geometry geometry;
    std::string geometry_name;
    //std::unordered_map<std::string, std::string> properties;
    std::array<float, 4> bbox;
    };

    struct CRS {
    std::string type;
    //std::unordered_map<std::string, std::string> properties;
    };

    struct WFSFeatureCollection {
    std::string type;
    std::vector<Feature> features;
    int totalFeatures;
    int numberMatched;
    int numberReturned;
    std::string timeStamp;
    CRS crs; 
    std::array<float, 4> bbox;
    };
}





#endif // CSP_WFS_OVERLAYS_WFS_TYPES_HPP