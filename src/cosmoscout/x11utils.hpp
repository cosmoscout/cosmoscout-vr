////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_X11UTILS_HPP
#define CS_X11UTILS_HPP

#include <string>

namespace x11utils {

/// Freeglut does not set the XClassHint which is required to properly show the application's name
/// in various places under X11 (for example in Gnome Shell's application switcher).
void setXClassHint(std::string const& title);

/// Freeglut does not support setting a window's icon on X11. If no .desktop file is used, this
/// results in empty icons in various places (for example in Gnome Shell's application switcher).
/// This uses XChangeProperty to set an icon.
void setAppIcon(std::string const& file);

} // namespace x11utils

#endif // CS_X11UTILS_HPP
