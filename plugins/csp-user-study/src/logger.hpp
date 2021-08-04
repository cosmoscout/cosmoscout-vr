////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_ANCHOR_LABELS_LOGGER_HPP
#define CSP_ANCHOR_LABELS_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::anchorlabels {

/// This creates the default singleton logger for "csp-anchor-labels" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::anchorlabels

#endif // CSP_ANCHOR_LABELS_LOGGER_HPP
