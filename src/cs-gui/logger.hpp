////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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
