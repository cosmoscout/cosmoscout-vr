////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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
