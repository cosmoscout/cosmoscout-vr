////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_VISTA_GUIITEM_HPP
#define CS_GUI_VISTA_GUIITEM_HPP

#include "WebView.hpp"

class VistaTexture;

namespace cs::gui {

/// GuiItem is an implementation of WebView specifically designed to be rendered in an OpenGL
/// context. It renders the HTML contents into an OpenGL texture for use in the rendering pipeline.
class CS_GUI_EXPORT GuiItem : public WebView {

 public:
  /// Creates a new GuiItem for the given page at the location of the URL.
  explicit GuiItem(std::string const& url, bool allowLocalFileAccess = false);
  virtual ~GuiItem();

  void setSizeX(unsigned int value); ///< Sets the width of the item in pixels.
  void setSizeY(unsigned int value); ///< Sets the height of the item in pixels.
  void setPositionX(int value);      ///< Sets the x position of the item in pixels.
  void setPositionY(int value);      ///< Sets the y position of the item in pixels.
  void setOffsetX(int value);        ///< Sets the x offset of the item in pixels.
  void setOffsetY(int value);        ///< Sets the y offset of the item in pixels.
  void setRelSizeX(float value);     ///< Sets the width of the item in screen space [0..1].
  void setRelSizeY(float value);     ///< Sets the height of the item in screen space [0..1].
  void setRelPositionX(float value); ///< Sets the x position of the item in screen space [0..1].
  void setRelPositionY(float value); ///< Sets the y position of the item in screen space [0..1].
  void setRelOffsetX(float value);   ///< Sets the x offset of the item in screen space [0..1].
  void setRelOffsetY(float value);   ///< Sets the y offset of the item in screen space [0..1].

  unsigned int getSizeX() const;        ///< Get the width of the item in pixels.
  unsigned int getSizeY() const;        ///< Get the height of the item in pixels.
  int          getPositionX() const;    ///< Get the x position of the item in pixels.
  int          getPositionY() const;    ///< Get the y position of the item in pixels.
  float        getRelSizeX() const;     ///< Get the x offset of the item in pixels.
  float        getRelSizeY() const;     ///< Get the y offset of the item in pixels.
  float        getRelPositionX() const; ///< Get the width of the item in screen space [0..1].
  float        getRelPositionY() const; ///< Get the height of the item in screen space [0..1].
  int          getOffsetX() const;      ///< Get the x position of the item in screen space [0..1].
  int          getOffsetY() const;      ///< Get the y position of the item in screen space [0..1].
  float        getRelOffsetX() const;   ///< Get the x offset of the item in screen space [0..1].
  float        getRelOffsetY() const;   ///< Get the y offset of the item in screen space [0..1].

  /// The enabled flag determines if the item will be rendered.
  void setIsEnabled(bool bEnabled);
  bool getIsEnabled() const;

  /// Calculates the position of the mouse within this items bounds.
  ///
  /// @param      areaX  The x position of the mouse in the parent GuiArea.
  /// @param      areaY  The y position of the mouse in the parent GuiArea.
  /// @param[out] x      The x position of the mouse relative to this GuiItem.
  /// @param[out] y      The y position of the mouse relative to this GuiItem.
  ///
  /// @return If the mouse is in this items bounds at all.
  bool calculateMousePosition(int areaX, int areaY, int& x, int& y);

  /// Gets called, when the parent GuiArea changes size.
  void onAreaResize(int width, int height);

  /// @return The current HTML output as an OpenGL texture.
  VistaTexture* getTexture() const;

 private:
  void updateTexture(DrawEvent const& event);
  void updateSizes();

  VistaTexture* mTexture;

  int mAreaWidth, mAreaHeight; // in pixels

  unsigned int mSizeX, mSizeY;               // in pixels
  int          mPositionX, mPositionY;       // in pixels
  int          mOffsetX, mOffsetY;           // in pixels
  float        mRelSizeX, mRelSizeY;         // in [0..1]
  float        mRelPositionX, mRelPositionY; // in [0..1]
  float        mRelOffsetX, mRelOffsetY;     // in [0..1]
  bool mIsRelSizeX, mIsRelSizeY, mIsRelPositionX, mIsRelPositionY, mIsRelOffsetX, mIsRelOffsetY;
  bool mIsEnabled = true;
};

} // namespace cs::gui

#endif // CS_GUI_VISTA_GUIITEM_HPP
