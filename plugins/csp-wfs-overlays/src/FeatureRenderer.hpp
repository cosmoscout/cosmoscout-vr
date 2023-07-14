////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WFS_OVERLAYS_FEATURE_RENDERER_HPP
#define CSP_WFS_OVERLAYS_FEATURE_RENDERER_HPP

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <vector>
#include <memory>
#include "WFSTypes.hpp"
#include <glm/glm.hpp>
#include "../../../src/cs-core/SolarSystem.hpp"

namespace csp::wfsoverlays { 
  class FeatureRenderer : public IVistaOpenGLDraw {
    public:
      template <typename T>
      FeatureRenderer (std::string mType, T coordinates, std::shared_ptr<cs::core::SolarSystem> solarSystem, std::shared_ptr<cs::core::Settings> settings); 
      ~FeatureRenderer();
      bool Do() override;
      bool GetBoundingBox(VistaBoundingBox& bb) override;
    private:
      std::vector<glm::vec3> mCoordinates;
      std::unique_ptr<VistaOpenGLNode> mGLNode;
      std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
      std::string mType;
      std::shared_ptr<cs::core::Settings> mSettings;
      int mHDRConnection;
      bool mShaderDirty;

      unsigned int VAO;
      unsigned int shaderProgram;

      static const char* FEATURE_VERT;
      static const char* FEATURE_FRAG;
  };
}


#endif // CSP_WFS_OVERLAYS_FEATURE_RENDERER_HPP