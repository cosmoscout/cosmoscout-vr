////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_KEY_EVENT_HPP
#define CS_GUI_KEY_EVENT_HPP

#include "cs_gui_export.hpp"
#include "types.hpp"

namespace cs::gui {

/// Contains the data contained in an interaction with the keyboard.
struct CS_GUI_EXPORT KeyEvent {

  enum class Type {
    ePress,     ///< A non character key was pressed.
    eRelease,   ///< A non character key was released.
    eCharacter, ///< A character key was pressed.
    eInvalid
  };

  KeyEvent();

  KeyEvent(int key, int mods);

  void setMods(int mods);

  /// PRESS, RELEASE, CHARACTER.
  Type mType{Type::ePress};

  /// Bitwise or of any Modifier defined in types.hpp.
  uint32_t mModifiers{};

  union {
    /// Only used for PRESS and RELEASE.
    Key mKey;

    /// Only used for CHARACTER.
    uint16_t mCharacter{};
  };
};

} // namespace cs::gui

#endif // CS_GUI_KEY_EVENT_HPP
