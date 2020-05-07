////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_VISTA_GUIAREA_HPP
#define CS_GUI_VISTA_GUIAREA_HPP

#include "cs_gui_export.hpp"

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
class VistaVertexArrayObject;
class VistaBufferObject;

namespace cs::gui {

class GuiItem;

/// A container which holds a collection of GuiItems.
class CS_GUI_EXPORT GuiArea {

 public:
  GuiArea() = default;

  GuiArea(GuiArea const& other) = delete;
  GuiArea(GuiArea&& other)      = delete;

  GuiArea& operator=(GuiArea const& other) = delete;
  GuiArea& operator=(GuiArea&& other) = delete;

  ~GuiArea() = default;

  virtual int getWidth() const  = 0;
  virtual int getHeight() const = 0;

  /// Adds an item to the area.
  void addItem(GuiItem* item, unsigned int index = 0);

  /// Removes the item from the area.
  void removeItem(GuiItem* item);

  /// Removes the item at the given index from the area.
  void removeItem(unsigned int index = 0);

  /// Returns the item at the given index.
  GuiItem* getItem(unsigned int index = 0);

  /// Returns the item at the given coordinates.
  ///
  /// @param checkAlpha            If true an item is only returned if it has an alpha value greater
  ///                              than zero at the given coordinates.
  /// @param excludeNoninteractive If true an item is only returned if it is interactive.
  /// @param excludeDisabled       If true an item is only returned if it is enabled.
  GuiItem* getItemAt(int x, int y, bool checkAlpha = true, bool excludeNoninteractive = true,
      bool excludeDisabled = true);

  std::vector<GuiItem*> const& getItems() const;

 protected:
  virtual void updateItems();

 private:
  std::vector<GuiItem*> mItems;
};

} // namespace cs::gui

#endif // CS_GUI_VISTA_GUIAREA_HPP
