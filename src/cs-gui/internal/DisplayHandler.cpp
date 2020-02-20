////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DisplayHandler.hpp"

#include <iostream>
#include <spdlog/spdlog.h>

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DisplayHandler::OnConsoleMessage(CefRefPtr<CefBrowser> browser, cef_log_severity_t level,
    CefString const& message, CefString const& source, int line) {

  std::string path(source.ToString());
  int         pos((int)path.find_last_of("/\\"));
  std::string logMessage("[" + path.substr(pos == std::string::npos ? 0 : pos + 1) + ":" +
                         std::to_string(line) + "] " + message.ToString());

  if (level == LOGSEVERITY_DEBUG) {
    spdlog::debug(logMessage);
  } else if (level == LOGSEVERITY_WARNING) {
    spdlog::warn(logMessage);
  } else if (level == LOGSEVERITY_ERROR) {
    spdlog::error(logMessage);
  } else if (level == LOGSEVERITY_FATAL) {
    spdlog::critical(logMessage);
  } else {
    spdlog::info(logMessage);
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
