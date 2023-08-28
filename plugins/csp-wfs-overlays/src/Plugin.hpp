////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WFS_OVERLAYS_PLUGIN_HPP
#define CSP_WFS_OVERLAYS_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"
#include <memory>
#include <unordered_set>
#include "LineRenderer.hpp"
#include "PointRenderer.hpp"
#include "WFSTypes.hpp"

namespace csp::wfsoverlays {

/// This plugin represents Web Feature Servivces data  in space. The plugin is configurable via the application
/// config file. See README.md for details.

struct InfoStruct { 
    glm::dvec2 mLongLatDegrees;
    glm::dvec2 mLongLatRadians; 
    glm::dvec3 mCartesian;
    double mOverSurfaceHeight;
    bool mHeightComesFromJson;
  };

class Plugin : public cs::core::PluginBase {
  
  public: 

    void init() override;
    void deInit() override;
    void update() override;

    void setWFSServer(std::string URL);
    void setWFSFeatureType(std::string featureType);
    void setRendering(double pointSize, double lineWidth);  
    double calculateDistance(InfoStruct const& p1, InfoStruct const& p2, glm::vec3 earthRadius);
    void correctHeight (InfoStruct const& struct1, InfoStruct const& struct2, InfoStruct& temporaryStruct, std::shared_ptr<const cs::scene::CelestialObject> earth);
    double calculateAngle (InfoStruct const& previousPoint, InfoStruct const& middlePoint, InfoStruct const& nextPoint);
    std::vector<InfoStruct> interpolation (std::vector<InfoStruct> const& structsIn, double thresholdAngle, glm::vec3 earthRadius, std::shared_ptr<const cs::scene::CelestialObject> earth);
    std::vector<glm::dvec3> generateMidPoint (std::vector <InfoStruct> const& structIn, float threshold, glm::vec3 earthRadius, std::shared_ptr<const cs::scene::CelestialObject> earth, glm::vec3 featureColor);     
  
  private:

    void onLoad();
    void onSave();

    std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();
    std::string mBaseUrl;
    WFSFeatureCollection mFeatureLocation; 
    DescribeFeatureType mPropertiesStruct;
    std::stringstream mJsonStream;
    std::unique_ptr<PointRenderer> mPointRenderer;
    std::unique_ptr<LineRenderer> mLineStringRenderer;
    std::unique_ptr<LineRenderer> mPolygonRenderer;
    std::string mColor;
    std::string mTime;
    std::string mSelectedFeature;
    
    int mOnLoadConnection       = -1;
    int mOnSaveConnection       = -1;
};

} // namespace csp::wfsoverlays

#endif // CSP_WFS_OVERLAYS_PLUGIN_HPP
