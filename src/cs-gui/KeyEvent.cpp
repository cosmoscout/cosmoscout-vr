////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "KeyEvent.hpp"

#include <VistaKernel/InteractionManager/VistaKeyboardSystemControl.h>

namespace cs::gui {

////////////////////////////////////////////////////////////////////////////////////////////////////

KeyEvent::KeyEvent()
    : mKey(Key::eUnknown) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

KeyEvent::KeyEvent(int key, int mods) {
  mType      = key < 0 ? KeyEvent::Type::eRelease : KeyEvent::Type::ePress;
  mCharacter = static_cast<uint16_t>(key);

  setMods(mods);

  switch (std::abs(key)) {
  case VISTA_KEY_F1:
    mKey = Key::eF1;
    return;
  case VISTA_KEY_F2:
    mKey = Key::eF2;
    return;
  case VISTA_KEY_F3:
    mKey = Key::eF3;
    return;
  case VISTA_KEY_F4:
    mKey = Key::eF4;
    return;
  case VISTA_KEY_F5:
    mKey = Key::eF5;
    return;
  case VISTA_KEY_F6:
    mKey = Key::eF6;
    return;
  case VISTA_KEY_F7:
    mKey = Key::eF7;
    return;
  case VISTA_KEY_F8:
    mKey = Key::eF8;
    return;
  case VISTA_KEY_F9:
    mKey = Key::eF9;
    return;
  case VISTA_KEY_F10:
    mKey = Key::eF10;
    return;
  case VISTA_KEY_F11:
    mKey = Key::eF11;
    return;
  case VISTA_KEY_F12:
    mKey = Key::eF12;
    return;
  case VISTA_KEY_LEFTARROW:
    mKey = Key::eLeft;
    return;
  case VISTA_KEY_UPARROW:
    mKey = Key::eUp;
    return;
  case VISTA_KEY_RIGHTARROW:
    mKey = Key::eRight;
    return;
  case VISTA_KEY_DOWNARROW:
    mKey = Key::eDown;
    return;
  case VISTA_KEY_PAGEUP:
    mKey = Key::ePageUp;
    return;
  case VISTA_KEY_PAGEDOWN:
    mKey = Key::ePageDown;
    return;
  case VISTA_KEY_HOME:
    mKey = Key::eHome;
    return;
  case VISTA_KEY_END:
    mKey = Key::eEnd;
    return;
  case VISTA_KEY_ESC:
    mKey = Key::eEscape;
    return;
  case VISTA_KEY_ENTER:
    mCharacter = '\n';
    mType      = key < 0 ? KeyEvent::Type::eInvalid : KeyEvent::Type::eCharacter;
    mKey       = Key::eReturn;
    return;
  case VISTA_KEY_TAB:
    mCharacter = '\t';
    mType      = key < 0 ? KeyEvent::Type::eInvalid : KeyEvent::Type::eCharacter;
    mKey       = Key::eTab;
    return;
  case VISTA_KEY_BACKSPACE:
    mKey = Key::eBackspace;
    return;
  case VISTA_KEY_DELETE:
    mKey = Key::eKeyDelete;
    return;
  case VISTA_KEY_SHIFT_LEFT:
    mKey = Key::eLeftShift;
    return;
  case VISTA_KEY_SHIFT_RIGHT:
    mKey = Key::eRightShift;
    return;
  case VISTA_KEY_CTRL_LEFT:
    mKey = Key::eLeftControl;
    return;
  case VISTA_KEY_CTRL_RIGHT:
    mKey = Key::eRightControl;
    return;
  case VISTA_KEY_ALT_LEFT:
  case VISTA_KEY_ALT_RIGHT:
    mKey = Key::eAlt;
    return;
  }

  if (key <= 0) {
    mType = KeyEvent::Type::eInvalid;
    return;
  }

  mType = KeyEvent::Type::eCharacter;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void KeyEvent::setMods(int mods) {
  mModifiers = 0;
  if (mods & VISTA_KEYMOD_SHIFT) {
    mModifiers |= int(Modifier::eShift);
  }

  if (mods & VISTA_KEYMOD_CTRL) {
    mModifiers |= int(Modifier::eControl);
  }

  if (mods & VISTA_KEYMOD_ALT) {
    mModifiers |= int(Modifier::eAlt);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
