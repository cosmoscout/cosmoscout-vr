////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_LOGGER_HPP
#define CSL_NODE_EDITOR_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csl::nodeeditor {

/// This creates the default singleton logger for "csl-node-editor" when called for the
/// first time and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_LOGGER_HPP
