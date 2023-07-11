////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_USER_STUDY_LOGGER_HPP
#define CSP_USER_STUDY_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::userstudy {

/// This creates the default singleton logger for "csp-user-study" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::userstudy

#endif // CSP_USER_STUDY_LOGGER_HPP
