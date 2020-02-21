////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_LOGGER_HPP
#define CS_GUI_LOGGER_HPP

#include "cs_gui_export.hpp"

namespace cs::gui::logger {

/// This creates the default logger for "cs-gui" and is called at startup by the main() method.
/// See ../cs-utils/logger.hpp for more logging details.
CS_GUI_EXPORT void init();

} // namespace cs::gui::logger

#endif // CS_GUI_LOGGER_HPP
