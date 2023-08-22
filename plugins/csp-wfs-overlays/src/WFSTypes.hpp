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
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "../../../src/cs-utils/Property.hpp"
#include "../../../src/cs-core/Settings.hpp"


namespace csp::wfsoverlays {
    struct Settings {
      cs::utils::DefaultProperty<bool> mEnabled{true};
      std::vector<std::string> mWfs; 
    };

    // TODO: Consider adding the         const std::string PointType = "Point";  

    // Defining the vectors we will later use (just for a tidy code)
    using PointCoordinates = std::vector<double>;
    using LineStringCoordinates = std::vector<std::vector<double>>; // Will work for MultiPoint
    using PolygonCoordinates = std::vector <std::vector <std::vector<double>>>; // Will work for MultiLineString
    using MultiPolygonCoordinates = std::vector <std::vector <std::vector <std::vector<double>>>>;

    struct GeometryBase {
        std::string mType;
        GeometryBase(std::string type) : mType(type) {}
        virtual ~GeometryBase() = default;
    }; 

    struct Point : public GeometryBase {
        PointCoordinates mCoordinates;
        Point (PointCoordinates coordinates) : GeometryBase("Point"), mCoordinates(coordinates) {}
    };

    struct MultiPoint : public GeometryBase {
        LineStringCoordinates mCoordinates;
        MultiPoint (LineStringCoordinates coordinates) : GeometryBase("MultiPoint"), mCoordinates(coordinates) {}
    };

    struct LineString : public GeometryBase {
        LineStringCoordinates mCoordinates;
        LineString (LineStringCoordinates coordinates) : GeometryBase("LineString"), mCoordinates(coordinates) {}
    };

    struct MultiLineString : public GeometryBase {
        PolygonCoordinates mCoordinates;
        MultiLineString (PolygonCoordinates coordinates) : GeometryBase("MultiLineString"), mCoordinates(coordinates) {}
    };

    struct Polygon : public GeometryBase {
        PolygonCoordinates mCoordinates;
        Polygon (PolygonCoordinates coordinates) : GeometryBase("Polygon"), mCoordinates(coordinates) {}
    } ;

    struct MultiPolygon : public GeometryBase {
        MultiPolygonCoordinates mCoordinates;
        MultiPolygon (MultiPolygonCoordinates coordinates) : GeometryBase("MultiPolygon"), mCoordinates(coordinates) {}
    };

    struct Feature {  
        std::string type;
        std::string id;
        std::shared_ptr<GeometryBase> geometry;
        std::string geometry_name;
        std::unordered_map<std::string, nlohmann::json> properties;
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

    /////////////////////////////

    struct Property {
        std::string name;
        int maxOccurs;
        int minOccurs;
        bool nillable;
        std::string type;
        std::string localType;
    };

    struct FeatureType {
        std::string typeName;
        std::vector<Property> properties;
    };

    struct DescribeFeatureType {
        std::string elementFormDefault;
        std::string targetNamespace;
        std::string targetPrefix;
        std::vector<FeatureType> featureTypes;
    };

    double mPointSize;
    double mLineWidth;

}


#endif // CSP_WFS_OVERLAYS_WFS_TYPES_HPP