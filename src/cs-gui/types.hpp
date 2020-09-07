////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_TYPES_HPP
#define CS_GUI_TYPES_HPP

#include "cs_gui_export.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace cs::gui {

/// Data describing the new state of a part of the GUI, when it changed.
struct CS_GUI_EXPORT DrawEvent {
  int            mX;       ///< The x coordinate of the redrawn area.
  int            mY;       ///< The y coordinate of the redrawn area.
  int            mWidth;   ///< The width of the redrawn area.
  int            mHeight;  ///< The height of the redrawn area.
  bool           mResized; ///< If the event was triggered by a resize.
  const uint8_t* mData;    ///< The new pixel data of the redrawn area.
};

using DrawCallback = std::function<uint8_t*(const DrawEvent&)>;

using JSType = std::variant<double, bool, std::string>;

/// The visual appearance of the cursor.
enum class CS_GUI_EXPORT Cursor : int {
  ePointer = 0,
  eCross,
  eHand,
  eIbeam,
  eWait,
  eHelp,
  eEastResize,
  eNorthResize,
  eNortheastResize,
  eNorthwestResize,
  eSouthResize,
  eSoutheastResize,
  eSouthwestResize,
  eWestResize,
  eNorthsouthResize,
  eEastwestResize,
  eNortheastsouthwestResize,
  eNorthwestsoutheastResize,
  eColumnResize,
  eRowResize,
  eMiddlePanning,
  eEastPanning,
  eNorthPanning,
  eNortheastPanning,
  eNorthwestPanning,
  eSouthPanning,
  eSoutheastPanning,
  eSouthwestPanning,
  eWestPanning,
  eMove,
  eVerticaltext,
  eCell,
  eContextmenu,
  eAlias,
  eProgress,
  eNodrop,
  eCopy,
  eNone,
  eNotallowed,
  eZoomin,
  eZoomout,
  eGrab,
  eGrabbing,
  eCustom
};

using CursorChangeCallback         = std::function<void(Cursor)>;
using RequestKeyboardFocusCallback = std::function<void(bool)>;

/// The mouse button that was interacted with.
enum class CS_GUI_EXPORT Button : int {
  eLeft    = 0,
  eMiddle  = 1,
  eRight   = 2,
  eButton4 = 3,
  eButton5 = 4,
  eButton6 = 5,
  eButton7 = 6,
  eButton8 = 7
};

/// Modifier keyboard keys that trigger side effects, when combined with other keys.
enum class CS_GUI_EXPORT Modifier : int {
  eNone         = 0,
  eCapsLock     = 1 << 0,
  eShift        = 1 << 1,
  eControl      = 1 << 2,
  eAlt          = 1 << 3,
  eLeftButton   = 1 << 4,
  eMiddleButton = 1 << 5,
  eRightButton  = 1 << 6,
  eCommand      = 1 << 7,
  eNumLock      = 1 << 8,
  eIsKeyPad     = 1 << 9,
  eIsLeft       = 1 << 10,
  eIsRight      = 1 << 11
};

/// Keyboard keys.
/// Codes based on the chromium key at
/// https://github.com/adobe/webkit/blob/master/Source/WebCore/platform/WindowsKeyboardCodes.h
enum class CS_GUI_EXPORT Key : uint16_t {
  eUnknown                = 0,
  eBackspace              = 0x08,
  eTab                    = 0x09,
  eClear                  = 0x0C,
  eReturn                 = 0x0D,
  eShift                  = 0x10,
  eControl                = 0x11,
  eAlt                    = 0x12,
  ePause                  = 0x13,
  eCapsLock               = 0x14,
  eKana                   = 0x15,
  eJunja                  = 0x17,
  eFinal                  = 0x18,
  eHanja                  = 0x19,
  eEscape                 = 0x1B,
  eConvert                = 0x1C,
  eNonconvert             = 0x1D,
  eAccept                 = 0x1E,
  eModechange             = 0x1F,
  eSpace                  = 0x20,
  ePageUp                 = 0x21,
  ePageDown               = 0x22,
  eEnd                    = 0x23,
  eHome                   = 0x24,
  eLeft                   = 0x25,
  eUp                     = 0x26,
  eRight                  = 0x27,
  eDown                   = 0x28,
  eSelect                 = 0x29,
  ePrint                  = 0x2A,
  eExecute                = 0x2B,
  ePrintScreen            = 0x2C,
  eInsert                 = 0x2D,
  eKeyDelete              = 0x2E,
  eHelp                   = 0x2F,
  eKey0                   = 0x30,
  eKey1                   = 0x31,
  eKey2                   = 0x32,
  eKey3                   = 0x33,
  eKey4                   = 0x34,
  eKey5                   = 0x35,
  eKey6                   = 0x36,
  eKey7                   = 0x37,
  eKey8                   = 0x38,
  eKey9                   = 0x39,
  eKeyA                   = 0x41,
  eKeyB                   = 0x42,
  eKeyC                   = 0x43,
  eKeyD                   = 0x44,
  eKeyE                   = 0x45,
  eKeyF                   = 0x46,
  eKeyG                   = 0x47,
  eKeyH                   = 0x48,
  eKeyI                   = 0x49,
  eKeyJ                   = 0x4A,
  eKeyK                   = 0x4B,
  eKeyL                   = 0x4C,
  eKeyM                   = 0x4D,
  eKeyN                   = 0x4E,
  eKeyO                   = 0x4F,
  eKeyP                   = 0x50,
  eKeyQ                   = 0x51,
  eKeyR                   = 0x52,
  eKeyS                   = 0x53,
  eKeyT                   = 0x54,
  eKeyU                   = 0x55,
  eKeyV                   = 0x56,
  eKeyW                   = 0x57,
  eKeyX                   = 0x58,
  eKeyY                   = 0x59,
  eKeyZ                   = 0x5A,
  eLeftSuper              = 0x5B,
  eRightSuper             = 0x5C,
  eApps                   = 0x5D,
  eSleep                  = 0x5F,
  eKp0                    = 0x60,
  eKp1                    = 0x61,
  eKp2                    = 0x62,
  eKp3                    = 0x63,
  eKp4                    = 0x64,
  eKp5                    = 0x65,
  eKp6                    = 0x66,
  eKp7                    = 0x67,
  eKp8                    = 0x68,
  eKp9                    = 0x69,
  eKpMultiply             = 0x6A,
  eKpAdd                  = 0x6B,
  eKpSeparator            = 0x6C,
  eKpSubtract             = 0x6D,
  eKpDecimal              = 0x6E,
  eKpDivide               = 0x6F,
  eF1                     = 0x70,
  eF2                     = 0x71,
  eF3                     = 0x72,
  eF4                     = 0x73,
  eF5                     = 0x74,
  eF6                     = 0x75,
  eF7                     = 0x76,
  eF8                     = 0x77,
  eF9                     = 0x78,
  eF10                    = 0x79,
  eF11                    = 0x7A,
  eF12                    = 0x7B,
  eF13                    = 0x7C,
  eF14                    = 0x7D,
  eF15                    = 0x7E,
  eF16                    = 0x7F,
  eF17                    = 0x80,
  eF18                    = 0x81,
  eF19                    = 0x82,
  eF20                    = 0x83,
  eF21                    = 0x84,
  eF22                    = 0x85,
  eF23                    = 0x86,
  eF24                    = 0x87,
  eNumLock                = 0x90,
  eScrollLock             = 0x91,
  eLeftShift              = 0xA0,
  eRightShift             = 0xA1,
  eLeftControl            = 0xA2,
  eRightControl           = 0xA3,
  eLeftMenu               = 0xA4,
  eRightMenu              = 0xA5,
  eBrowserBack            = 0xA6,
  eBrowserForward         = 0xA7,
  eBrowserRefresh         = 0xA8,
  eBrowserStop            = 0xA9,
  eBrowserSearch          = 0xAA,
  eBrowserFavorites       = 0xAB,
  eBrowserHome            = 0xAC,
  eVolumeMute             = 0xAD,
  eVolumeDown             = 0xAE,
  eVolumeUp               = 0xAF,
  eMediaNextTrack         = 0xB0,
  eMediaPrevTrack         = 0xB1,
  eMediaStop              = 0xB2,
  eMediaPlayPause         = 0xB3,
  eMediaLaunchMail        = 0xB4,
  eMediaLaunchMediaSelect = 0xB5,
  eMediaLaunchApp1        = 0xB6,
  eMediaLaunchApp2        = 0xB7,
  ePlus                   = 0xBB,
  eComma                  = 0xBC,
  eMinus                  = 0xBD,
  ePeriod                 = 0xBE,
  eOem1                   = 0xBA, ///< The ';:' key
  eOem2                   = 0xBF, ///< The '/?' key
  eOem3                   = 0xC0, ///< The '`~' key
  eOem4                   = 0xDB, ///< The '[{' key
  eOem5                   = 0xDC, ///< The '\|' key
  eOem6                   = 0xDD, ///< The ']}' key
  eOem7                   = 0xDE, ///< The 'single-quote/double-quote' key
  eOem8                   = 0xDF, ///< Used for miscellaneous characters; it can vary by keyboard.
  eOem102                 = 0xE2, ///< Either the angle bracket key or the backslash key on the
                                  ///< RT 102-key keyboard
};

/// Appends the enum name as written in the enum definition.
CS_GUI_EXPORT std::ostream& operator<<(std::ostream& os, Key key);

namespace detail {

class Browser;
class WebApp;
class WebViewClient;

} // namespace detail

} // namespace cs::gui

#endif // CS_GUI_TYPES_HPP
