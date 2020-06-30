////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_LOGGER_HPP
#define CS_GUI_LOGGER_HPP

#include "cs_gui_export.hpp"

#include <spdlog/spdlog.h>

namespace cs::gui {

/// This creates the default singleton logger for "cs-gui" when called for the first time and
/// returns it. See cs-utils/logger.hpp for more logging details.
CS_GUI_EXPORT spdlog::logger& logger();

} // namespace cs::gui

#endif // CS_GUI_LOGGER_HPP
