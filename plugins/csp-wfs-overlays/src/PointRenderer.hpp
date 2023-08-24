////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WFS_OVERLAYS_POINT_RENDERER_HPP
#define CSP_WFS_OVERLAYS_POINT_RENDERER_HPP

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <vector>
#include <memory>
#include "WFSTypes.hpp"
#include <glm/glm.hpp>
#include "../../../src/cs-core/SolarSystem.hpp"
#include <VistaOGLExt/VistaTexture.h>

namespace csp::wfsoverlays { 
  class PointRenderer : public IVistaOpenGLDraw {

    public:

      PointRenderer (std::vector<glm::vec3> coordinates, std::shared_ptr<cs::core::SolarSystem> solarSystem, std::shared_ptr<cs::core::Settings> settings, 
                    double pointSize, std::shared_ptr<Settings> pluginSettings);    
      ~PointRenderer();
      bool Do() override;
      bool GetBoundingBox(VistaBoundingBox& bb) override;

    private:

      std::vector<glm::vec3> mCoordinates;
      std::unique_ptr<VistaOpenGLNode> mGLNode;
      std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
      std::shared_ptr<cs::core::Settings> mSettings;
      std::shared_ptr<Settings> mPluginSettings;
      std::unique_ptr<VistaTexture> mTexture; 
      double mPointSizeInput;
      int mHDRConnection;
      bool mShaderDirty;
      
      unsigned int mVAO;
      unsigned int mShaderProgram;

      static const char* FEATURE_VERT;
      static const char* FEATURE_GEOM;
      static const char* FEATURE_FRAG;
  };
}


#endif // CSP_WFS_OVERLAYS_POINT_RENDERER_HPP