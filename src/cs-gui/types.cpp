////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "types.hpp"

#include <iostream>

namespace cs::gui {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os, Key key) {
  switch (key) {
  case Key::eUnknown:
    os << "eUnknown";
    break;
  case Key::eBackspace:
    os << "eBackspace";
    break;
  case Key::eTab:
    os << "eTab";
    break;
  case Key::eClear:
    os << "eClear";
    break;
  case Key::eReturn:
    os << "eReturn";
    break;
  case Key::eShift:
    os << "eShift";
    break;
  case Key::eControl:
    os << "eControl";
    break;
  case Key::eAlt:
    os << "eAlt";
    break;
  case Key::ePause:
    os << "ePause";
    break;
  case Key::eCapsLock:
    os << "eCapsLock";
    break;
  case Key::eKana:
    os << "eKana";
    break;
  case Key::eJunja:
    os << "eJunja";
    break;
  case Key::eFinal:
    os << "eFinal";
    break;
  case Key::eHanja:
    os << "eHanja";
    break;
  case Key::eEscape:
    os << "eEscape";
    break;
  case Key::eConvert:
    os << "eConvert";
    break;
  case Key::eNonconvert:
    os << "eNonconvert";
    break;
  case Key::eAccept:
    os << "eAccept";
    break;
  case Key::eModechange:
    os << "eModechange";
    break;
  case Key::eSpace:
    os << "eSpace";
    break;
  case Key::ePageUp:
    os << "ePageUp";
    break;
  case Key::ePageDown:
    os << "ePageDown";
    break;
  case Key::eEnd:
    os << "eEnd";
    break;
  case Key::eHome:
    os << "eHome";
    break;
  case Key::eLeft:
    os << "eLeft";
    break;
  case Key::eUp:
    os << "eUp";
    break;
  case Key::eRight:
    os << "eRight";
    break;
  case Key::eDown:
    os << "eDown";
    break;
  case Key::eSelect:
    os << "eSelect";
    break;
  case Key::ePrint:
    os << "ePrint";
    break;
  case Key::eExecute:
    os << "eExecute";
    break;
  case Key::ePrintScreen:
    os << "ePrintScreen";
    break;
  case Key::eInsert:
    os << "eInsert";
    break;
  case Key::eKeyDelete:
    os << "eKeyDelete";
    break;
  case Key::eHelp:
    os << "eHelp";
    break;
  case Key::eKey0:
    os << "eKey0";
    break;
  case Key::eKey1:
    os << "eKey1";
    break;
  case Key::eKey2:
    os << "eKey2";
    break;
  case Key::eKey3:
    os << "eKey3";
    break;
  case Key::eKey4:
    os << "eKey4";
    break;
  case Key::eKey5:
    os << "eKey5";
    break;
  case Key::eKey6:
    os << "eKey6";
    break;
  case Key::eKey7:
    os << "eKey7";
    break;
  case Key::eKey8:
    os << "eKey8";
    break;
  case Key::eKey9:
    os << "eKey9";
    break;
  case Key::eKeyA:
    os << "eKeyA";
    break;
  case Key::eKeyB:
    os << "eKeyB";
    break;
  case Key::eKeyC:
    os << "eKeyC";
    break;
  case Key::eKeyD:
    os << "eKeyD";
    break;
  case Key::eKeyE:
    os << "eKeyE";
    break;
  case Key::eKeyF:
    os << "eKeyF";
    break;
  case Key::eKeyG:
    os << "eKeyG";
    break;
  case Key::eKeyH:
    os << "eKeyH";
    break;
  case Key::eKeyI:
    os << "eKeyI";
    break;
  case Key::eKeyJ:
    os << "eKeyJ";
    break;
  case Key::eKeyK:
    os << "eKeyK";
    break;
  case Key::eKeyL:
    os << "eKeyL";
    break;
  case Key::eKeyM:
    os << "eKeyM";
    break;
  case Key::eKeyN:
    os << "eKeyN";
    break;
  case Key::eKeyO:
    os << "eKeyO";
    break;
  case Key::eKeyP:
    os << "eKeyP";
    break;
  case Key::eKeyQ:
    os << "eKeyQ";
    break;
  case Key::eKeyR:
    os << "eKeyR";
    break;
  case Key::eKeyS:
    os << "eKeyS";
    break;
  case Key::eKeyT:
    os << "eKeyT";
    break;
  case Key::eKeyU:
    os << "eKeyU";
    break;
  case Key::eKeyV:
    os << "eKeyV";
    break;
  case Key::eKeyW:
    os << "eKeyW";
    break;
  case Key::eKeyX:
    os << "eKeyX";
    break;
  case Key::eKeyY:
    os << "eKeyY";
    break;
  case Key::eKeyZ:
    os << "eKeyZ";
    break;
  case Key::eLeftSuper:
    os << "eLeftSuper";
    break;
  case Key::eRightSuper:
    os << "eRightSuper";
    break;
  case Key::eApps:
    os << "eApps";
    break;
  case Key::eSleep:
    os << "eSleep";
    break;
  case Key::eKp0:
    os << "eKp0";
    break;
  case Key::eKp1:
    os << "eKp1";
    break;
  case Key::eKp2:
    os << "eKp2";
    break;
  case Key::eKp3:
    os << "eKp3";
    break;
  case Key::eKp4:
    os << "eKp4";
    break;
  case Key::eKp5:
    os << "eKp5";
    break;
  case Key::eKp6:
    os << "eKp6";
    break;
  case Key::eKp7:
    os << "eKp7";
    break;
  case Key::eKp8:
    os << "eKp8";
    break;
  case Key::eKp9:
    os << "eKp9";
    break;
  case Key::eKpMultiply:
    os << "eKpMultiply";
    break;
  case Key::eKpAdd:
    os << "eKpAdd";
    break;
  case Key::eKpSeparator:
    os << "eKpSeparator";
    break;
  case Key::eKpSubtract:
    os << "eKpSubtract";
    break;
  case Key::eKpDecimal:
    os << "eKpDecimal";
    break;
  case Key::eKpDivide:
    os << "eKpDivide";
    break;
  case Key::eF1:
    os << "eF1";
    break;
  case Key::eF2:
    os << "eF2";
    break;
  case Key::eF3:
    os << "eF3";
    break;
  case Key::eF4:
    os << "eF4";
    break;
  case Key::eF5:
    os << "eF5";
    break;
  case Key::eF6:
    os << "eF6";
    break;
  case Key::eF7:
    os << "eF7";
    break;
  case Key::eF8:
    os << "eF8";
    break;
  case Key::eF9:
    os << "eF9";
    break;
  case Key::eF10:
    os << "eF10";
    break;
  case Key::eF11:
    os << "eF11";
    break;
  case Key::eF12:
    os << "eF12";
    break;
  case Key::eF13:
    os << "eF13";
    break;
  case Key::eF14:
    os << "eF14";
    break;
  case Key::eF15:
    os << "eF15";
    break;
  case Key::eF16:
    os << "eF16";
    break;
  case Key::eF17:
    os << "eF17";
    break;
  case Key::eF18:
    os << "eF18";
    break;
  case Key::eF19:
    os << "eF19";
    break;
  case Key::eF20:
    os << "eF20";
    break;
  case Key::eF21:
    os << "eF21";
    break;
  case Key::eF22:
    os << "eF22";
    break;
  case Key::eF23:
    os << "eF23";
    break;
  case Key::eF24:
    os << "eF24";
    break;
  case Key::eNumLock:
    os << "eNumLock";
    break;
  case Key::eScrollLock:
    os << "eScrollLock";
    break;
  case Key::eLeftShift:
    os << "eLeftShift";
    break;
  case Key::eRightShift:
    os << "eRightShift";
    break;
  case Key::eLeftControl:
    os << "eLeftControl";
    break;
  case Key::eRightControl:
    os << "eRightControl";
    break;
  case Key::eLeftMenu:
    os << "eLeftMenu";
    break;
  case Key::eRightMenu:
    os << "eRightMenu";
    break;
  case Key::eBrowserBack:
    os << "eBrowserBack";
    break;
  case Key::eBrowserForward:
    os << "eBrowserForward";
    break;
  case Key::eBrowserRefresh:
    os << "eBrowserRefresh";
    break;
  case Key::eBrowserStop:
    os << "eBrowserStop";
    break;
  case Key::eBrowserSearch:
    os << "eBrowserSearch";
    break;
  case Key::eBrowserFavorites:
    os << "eBrowserFavorites";
    break;
  case Key::eBrowserHome:
    os << "eBrowserHome";
    break;
  case Key::eVolumeMute:
    os << "eVolumeMute";
    break;
  case Key::eVolumeDown:
    os << "eVolumeDown";
    break;
  case Key::eVolumeUp:
    os << "eVolumeUp";
    break;
  case Key::eMediaNextTrack:
    os << "eMediaNextTrack";
    break;
  case Key::eMediaPrevTrack:
    os << "eMediaPrevTrack";
    break;
  case Key::eMediaStop:
    os << "eMediaStop";
    break;
  case Key::eMediaPlayPause:
    os << "eMediaPlayPause";
    break;
  case Key::eMediaLaunchMail:
    os << "eMediaLaunchMail";
    break;
  case Key::eMediaLaunchMediaSelect:
    os << "eMediaLaunchMediaSelect";
    break;
  case Key::eMediaLaunchApp1:
    os << "eMediaLaunchApp1";
    break;
  case Key::eMediaLaunchApp2:
    os << "eMediaLaunchApp2";
    break;
  case Key::ePlus:
    os << "ePlus";
    break;
  case Key::eComma:
    os << "eComma";
    break;
  case Key::eMinus:
    os << "eMinus";
    break;
  case Key::ePeriod:
    os << "ePeriod";
    break;
  case Key::eOem1:
    os << "eOem1";
    break;
  case Key::eOem2:
    os << "eOem2";
    break;
  case Key::eOem3:
    os << "eOem3";
    break;
  case Key::eOem4:
    os << "eOem4";
    break;
  case Key::eOem5:
    os << "eOem5";
    break;
  case Key::eOem6:
    os << "eOem6";
    break;
  case Key::eOem7:
    os << "eOem7";
    break;
  case Key::eOem8:
    os << "eOem8";
    break;
  case Key::eOem102:
    os << "eOem102";
    break;
  }
  return os;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
