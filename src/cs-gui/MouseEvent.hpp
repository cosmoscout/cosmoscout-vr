////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_MOUSE_EVENT_HPP
#define CS_GUI_MOUSE_EVENT_HPP

#include "cs_gui_export.hpp"
#include "types.hpp"

namespace cs::gui {

/// Contains all information, when the state of the mouse changes.
struct CS_GUI_EXPORT MouseEvent {

  enum class Type {
    eMove,    ///< The mouse moved.
    eScroll,  ///< The scroll wheel moved.
    ePress,   ///< A mouse button was pressed.
    eRelease, ///< A mouse button was released.
    eLeave    ///< The mouse left the window
  };

  MouseEvent() // NOLINT(cppcoreguidelines-pro-type-member-init): Can't init both union types.
      : mX(0) {
  }

  /// Either eMove, eScroll, ePress or eRelease.
  Type mType{Type::eMove};

  union {
    /// X-position for eMove, x-direction for eScroll.
    int mX;

    /// Only used for ePress and eRelease.
    Button mButton;
  };

  /// Y-position for eMove, y-direction for eScroll.
  int mY{0};
};

} // namespace cs::gui

#endif // CS_GUI_MOUSE_EVENT_HPP
