////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VIRTUAL_SATELLITE_BOX_RENDERER_HPP
#define CSP_VIRTUAL_SATELLITE_BOX_RENDERER_HPP

#include "Plugin.hpp"
#include "RenderTypes.hpp"

#include <VistaBase/VistaColor.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaGLSLShader.h>

namespace csp::virtualsatellite {

class BoxRenderer : public IVistaOpenGLDraw {
 public:
  BoxRenderer(std::shared_ptr<Plugin::Settings> pluginSettings,
      std::shared_ptr<cs::core::SolarSystem>    solarSystem);

  BoxRenderer(BoxRenderer const& other) = delete;
  BoxRenderer(BoxRenderer&& other)      = default;

  BoxRenderer& operator=(BoxRenderer const& other) = delete;
  BoxRenderer& operator=(BoxRenderer&& other)      = default;

  ~BoxRenderer() override;

  void setObjectName(std::string objectName);
  void setBoxes(std::vector<Box> const& boxes);

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<Plugin::Settings>      mPluginSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::string                            mObjectName;
  std::vector<Box>                       mSolidBoxes;
  std::vector<Box>                       mTranslucentBoxes;

  VistaGLSLShader                  mShader;
  std::unique_ptr<VistaOpenGLNode> mGLNode;

  struct {
    uint32_t mvp   = 0;
    uint32_t color = 0;
  } mUniforms;

  uint32_t mVAO = 0;

  static const char* BOX_VERT;
  static const char* BOX_FRAG;
};

} // namespace csp::virtualsatellite

#endif // CSP_VIRTUAL_SATELLITE_BOX_RENDERER_HPP
