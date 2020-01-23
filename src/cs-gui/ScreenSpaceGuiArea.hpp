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
  ~ScreenSpaceGuiArea() override;

  int getWidth() const override;
  int getHeight() const override;

  void setSmooth(bool enable) override;

  /// Draws the UI to screen.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

  /// Handles changes to the screen size.
  void ObserverUpdate(IVistaObserveable* pObserveable, int nMsg, int nTicket) override;

 private:
  virtual void onViewportChange();

  VistaViewport*   mViewport;
  VistaGLSLShader* mShader                = nullptr;
  bool             mShaderDirty           = true;
  int              mWidth                 = 0;
  int              mHeight                = 0;
  int              mOldWidth              = 0;
  int              mOldHeight             = 0;
  int              mDelayedViewportUpdate = 0;
  bool             mSmooth                = false;
};

} // namespace cs::gui

#endif // CS_GUI_VISTA_SCREENSPACEGUIAREA_HPP
