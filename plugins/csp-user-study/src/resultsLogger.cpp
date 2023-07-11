////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "resultsLogger.hpp"

#include "../../../src/cs-utils/utils.hpp"

#include <chrono>
#include <ctime>
#include <spdlog/sinks/basic_file_sink.h>

namespace csp::userstudy {

////////////////////////////////////////////////////////////////////////////////////////////////////

spdlog::logger& resultsLogger() {
  // Get current date
  time_t     rawtime  = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  struct tm* timeinfo = nullptr;
  char       buffer[80];

  time(&rawtime);

  // Disables a warning in MSVC about using localtime_s, which isn't supported in GCC.
  CS_WARNINGS_PUSH
  CS_DISABLE_MSVC_WARNING(4996)

  timeinfo = localtime(&rawtime);

  CS_WARNINGS_POP

  strftime(buffer, sizeof(buffer), "%d-%m-%Y_%H-%M-%S", timeinfo);
  std::string date(buffer);

  // create sink with date in filename
  // TODO: uncomment date
  static auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      date +  "_userstudy_results_.log", true);

  static auto logger = std::make_unique<spdlog::logger>("results-logger", sink);
  logger->set_pattern("%^[%d.%m.%Y %H:%M:%S.%e]%$ %v"); // NOLINT(clang-analyzer-cplusplus.Move)
  logger->set_level(spdlog::level::trace);
  logger->flush_on(spdlog::level::info);
  return *logger;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::userstudy
