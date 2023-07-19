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
#include "FeatureRenderer.hpp"
#include "WFSTypes.hpp"

namespace csp::wfsoverlays {

/// This plugin represents Web Feature Servivces data  in space. The plugin is configurable via the application
/// config file. See README.md for details.
struct InfoStruct { 
    glm::dvec2 longLatDegrees;
    glm::dvec2 longLatRadians; 
    glm::dvec3 Cartesian;
    double overSurfaceHeight;
    bool heightComesFromJson;
  };
class Plugin : public cs::core::PluginBase {
  
  public:
    struct Settings {
      cs::utils::DefaultProperty<bool> mEnabled{true};
      std::vector<std::string> mWfs; 
    };
  
    void init() override;
    void deInit() override;
    void update() override;

    void setWFSServer(std::string URL);
    void setWFSFeatureType(std::string featureType);
    double calculateDistance(InfoStruct const& p1, InfoStruct const& p2, glm::vec3 earthRadius);

    std::vector<glm::dvec3> generateMidPoint (std::vector <InfoStruct> const& structIn, float threshold, 
                                                        glm::vec3 earthRadius, std::shared_ptr<const cs::scene::CelestialObject> earth);       // TODO: I think the Plugin:: isnt needed here

    
  private:

    void onLoad();
    void onSave();

    std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();
    std::string mBaseUrl;

    std::unique_ptr<FeatureRenderer> mPointRenderer;
    std::unique_ptr<FeatureRenderer> mLineStringRenderer;
    std::unique_ptr<FeatureRenderer> mPolygonRenderer;


    int mOnLoadConnection       = -1;
    int mOnSaveConnection       = -1;
};



} // namespace csp::wfsoverlays

#endif // CSP_WFS_OVERLAYS_PLUGIN_HPP
