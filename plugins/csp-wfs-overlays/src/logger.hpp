////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WFS_OVERLAYS_HPP
#define CSP_WFS_OVERLAYS_HPP

#include <spdlog/spdlog.h>

namespace csp::wfsoverlays {

/// This creates the default singleton logger for "csp-wfs-overlays" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::wfsoverlays

#endif // CSP_WFS_OVERLAYS_HPP
