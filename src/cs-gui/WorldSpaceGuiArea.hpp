////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_VISTA_WORLDSPACEGUIAREA_HPP
#define CS_GUI_VISTA_WORLDSPACEGUIAREA_HPP

#include "GuiArea.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

class VistaTransformMatrix;
class VistaVector3D;
class VistaQuaternion;
class VistaGLSLShader;

namespace cs::gui {

class GuiItem;

/// Responsible for drawing UI elements which are located in 3D space. For example a label
/// following a satellite.
class CS_GUI_EXPORT WorldSpaceGuiArea : public GuiArea, public IVistaOpenGLDraw {

 public:
  explicit WorldSpaceGuiArea(int width = 640, int height = 480);

  WorldSpaceGuiArea(WorldSpaceGuiArea const& other) = delete;
  WorldSpaceGuiArea(WorldSpaceGuiArea&& other)      = delete;

  WorldSpaceGuiArea& operator=(WorldSpaceGuiArea const& other) = delete;
  WorldSpaceGuiArea& operator=(WorldSpaceGuiArea&& other) = delete;

  ~WorldSpaceGuiArea() override = default;

  void setWidth(int width);
  void setHeight(int height);

  int getWidth() const override;
  int getHeight() const override;

  /// If true, the elements won't be occluded by world objects.
  void setIgnoreDepth(bool ignore);
  bool getIgnoreDepth() const;

  /// Determines whether to write the fragment depth value or not.
  bool getUseLinearDepthBuffer() const;
  void setUseLinearDepthBuffer(bool bEnable);

  /// Calculates the position of the mouse in pixels. vRayOrigin and vRayEnd should be in
  /// gui-plane-coordinates. The gui plane is the xy-plane with the normal pointing in positive
  /// z-direction. The result can be out of bounds of this area  - in this case 'false' is returned,
  /// but the values are compted anyways.
  virtual bool calculateMousePosition(
      VistaVector3D const& vRayOrigin, VistaVector3D const& vRayEnd, int& x, int& y);

  /// Draws the GuiElements to the screen.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& oBoundingBox) override;

 private:
  VistaGLSLShader mShader;
  bool            mShaderDirty          = true;
  bool            mIgnoreDepth          = false;
  bool            mUseLinearDepthBuffer = false;
  int             mWidth                = 0;
  int             mHeight               = 0;

  struct {
    uint32_t farClip          = 0;
    uint32_t projectionMatrix = 0;
    uint32_t modelViewMatrix  = 0;
    uint32_t texSize          = 0;
    uint32_t texture          = 0;
  } mUniforms;

  static const char* const QUAD_VERT;
  static const char* const QUAD_FRAG;
};

} // namespace cs::gui

#endif // CS_GUI_VISTA_WORLDSPACEGUIAREA_HPP
