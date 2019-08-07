////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_VISTA_SPRITEGUI_HPP
#define CS_GUI_VISTA_SPRITEGUI_HPP

#include "WebView.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

class VistaGLSLShader;
class VistaQuaternion;
class VistaTexture;
class VistaTransformMatrix;
class VistaVector3D;

namespace cs::gui {

/// DocTODO Not sure for what this is used.
class CS_GUI_EXPORT SpriteGui : public WebView, public IVistaOpenGLDraw {

 public:
  SpriteGui(std::string const& url, int width, int height);
  ~SpriteGui() override;

  void  setDepthOffset(float offset);
  float getDepthOffset() const;

  bool getUseLinearDepthBuffer() const;
  void setUseLinearDepthBuffer(bool bEnable);

  VistaTexture* getTexture() const;

  // IVistaOpenGLDraw interface --------------------------------------------------------------------
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  void updateTexture(DrawEvent const& event);

  VistaTexture*    mTexture;
  bool             mShaderDirty = true;
  VistaGLSLShader* mShader;
  bool             mUseLinearDepthBuffer = false;
  float            mDepthOffset          = 0.0;
};

} // namespace cs::gui

#endif // CS_GUI_VISTA_SPRITEGUI_HPP
