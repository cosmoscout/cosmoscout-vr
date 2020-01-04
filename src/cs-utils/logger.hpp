////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_LOGGER_HPP
#define CS_UTILS_LOGGER_HPP

#include "cs_utils_export.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace cs::utils::logger {

CS_UTILS_EXPORT void init();

inline void init(std::string const& name) {
  auto logger = spdlog::stdout_color_mt(name);
  logger->set_level(spdlog::level::debug);
  logger->set_pattern("%^[%L][%n]%$ %v");
  spdlog::set_default_logger(logger);
}

} // namespace cs::utils::logger

#endif // CS_UTILS_LOGGER_HPP
