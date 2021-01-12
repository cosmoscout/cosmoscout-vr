////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_VISTA_SCREENSPACEGUIAREA_HPP
#define CS_GUI_VISTA_SCREENSPACEGUIAREA_HPP

#include "GuiArea.hpp"

#include <VistaAspects/VistaObserver.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <vector>

class VistaTransformNode;
class VistaTransformMatrix;
class VistaVector3D;
class VistaQuaternion;
class VistaProjection;
class VistaViewport;
class VistaGLSLShader;
class VistaVertexArrayObject;
class VistaBufferObject;

namespace cs::gui {

/// This class is used to render static UI elements, which are always at the same position of the
/// screen.
class CS_GUI_EXPORT ScreenSpaceGuiArea : public GuiArea,
                                         public IVistaOpenGLDraw,
                                         public IVistaObserver {

 public:
  explicit ScreenSpaceGuiArea(VistaViewport* pViewport);

  ScreenSpaceGuiArea(ScreenSpaceGuiArea const& other) = delete;
  ScreenSpaceGuiArea(ScreenSpaceGuiArea&& other)      = delete;

  ScreenSpaceGuiArea& operator=(ScreenSpaceGuiArea const& other) = delete;
  ScreenSpaceGuiArea& operator=(ScreenSpaceGuiArea&& other) = delete;

  ~ScreenSpaceGuiArea() override = default;

  int getWidth() const override;
  int getHeight() const override;

  /// Draws the UI to screen.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& oBoundingBox) override;

  /// Handles changes to the screen size.
  void ObserverUpdate(IVistaObserveable* pObserveable, int nMsg, int nTicket) override;

 private:
  virtual void onViewportChange();

  VistaViewport*  mViewport;
  VistaGLSLShader mShader;
  bool            mShaderDirty           = true;
  int             mWidth                 = 0;
  int             mHeight                = 0;
  int             mDelayedViewportUpdate = 0;

  struct {
    uint32_t position = 0;
    uint32_t scale    = 0;
    uint32_t texSize  = 0;
    uint32_t texture  = 0;
  } mUniforms;

  static const char* const QUAD_VERT;
  static const char* const QUAD_FRAG;
};

} // namespace cs::gui

#endif // CS_GUI_VISTA_SCREENSPACEGUIAREA_HPP
