////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_LOGGER_HPP
#define CSP_LOD_BODIES_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::lodbodies {

/// This creates the default singleton logger for "csp-lod-bodies" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_LOGGER_HPP
