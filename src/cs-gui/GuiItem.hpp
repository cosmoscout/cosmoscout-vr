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

  GuiItem(GuiItem const& other) = delete;
  GuiItem(GuiItem&& other)      = delete;

  GuiItem& operator=(GuiItem const& other) = delete;
  GuiItem& operator=(GuiItem&& other) = delete;

  ~GuiItem() override;

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

  /// Returns the current width and height of the internal texture. This may differ from getSizeX()
  /// and getSizeY() as the texture is updated asynchronously and therfore it may take some frames
  /// to reflect size changes.
  int getTextureSizeX() const;
  int getTextureSizeY() const;

  /// The enabled flag determines if the item will be rendered.
  void setIsEnabled(bool bEnabled);
  bool getIsEnabled() const;

  /// Returns true when an HTML element is focused which can receive keyboard input.
  bool getIsKeyboardInputElementFocused() const;

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
  uint32_t getTexture() const;

 private:
  uint8_t* updateTexture(DrawEvent const& event);
  void     updateSizes();

  uint32_t mTextureBuffer{};
  uint32_t mTexture{};
  uint8_t* mBufferData = nullptr;

  // in pixels
  int mTextureSizeX = 0;
  int mTextureSizeY = 0;
  int mAreaWidth    = 0;
  int mAreaHeight   = 0;

  unsigned int mSizeX, mSizeY;               // in pixels
  int          mPositionX, mPositionY;       // in pixels
  int          mOffsetX, mOffsetY;           // in pixels
  float        mRelSizeX, mRelSizeY;         // in [0..1]
  float        mRelPositionX, mRelPositionY; // in [0..1]
  float        mRelOffsetX, mRelOffsetY;     // in [0..1]
  bool mIsRelSizeX, mIsRelSizeY, mIsRelPositionX, mIsRelPositionY, mIsRelOffsetX, mIsRelOffsetY;
  bool mIsEnabled                     = true;
  bool mIsKeyboardInputElementFocused = false;
};

} // namespace cs::gui

#endif // CS_GUI_VISTA_GUIITEM_HPP
