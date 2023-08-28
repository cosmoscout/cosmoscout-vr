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
        cs::utils::DefaultProperty<bool> mInterpolation{false};
    };

    // defining the vectors we will later use (just for a tidy code)
    using PointCoordinates = std::vector<double>;                                                           // for points
    using LineStringCoordinates = std::vector<std::vector<double>>;                                         // for lineStrings and multiPoints
    using PolygonCoordinates = std::vector <std::vector <std::vector<double>>>;                             // for polygons MultiLineString
    using MultiPolygonCoordinates = std::vector <std::vector <std::vector <std::vector<double>>>>;          // for multiPolygons

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
        std::string mType;
        std::string mId;
        std::shared_ptr<GeometryBase> mGeometry;
        std::string mGeometry_name;
        std::unordered_map<std::string, nlohmann::json> mProperties;
        // std::array<float, 4> mBbox;
    };

    struct CRS {
        std::string mType;
        //std::unordered_map<std::string, std::string> mProperties;
    };

    struct WFSFeatureCollection {
        std::string mType;
        std::vector<Feature> mFeatures;
        int mTotalFeatures;
        int mNumberMatched;
        int mNumberReturned;
        std::string mTimeStamp;
        CRS mCrs; 
        // std::array<float, 4> mBbox;
    };

    /////////////////////////////

    struct Property {
        std::string mName;
        int mMaxOccurs;
        int mMinOccurs;
        bool mNillable;
        std::string mType;
        std::string mLocalType;
    };

    struct FeatureType {
        std::string mTypeName;
        std::vector<Property> mProperties;
    };

    struct DescribeFeatureType {
        std::string mElementFormDefault;
        std::string mTargetNamespace;
        std::string mTargetPrefix;
        std::vector<FeatureType> mFeatureTypes;
    };

    double mPointSize;
    double mLineWidth;

}


#endif // CSP_WFS_OVERLAYS_WFS_TYPES_HPP