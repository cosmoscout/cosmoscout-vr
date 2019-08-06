////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DisplayHandler.hpp"

#include <iostream>

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DisplayHandler::OnConsoleMessage(
    CefRefPtr<CefBrowser> browser, CefString const& message, CefString const& source, int line) {

  std::string path(source.ToString());
  int         pos((int)path.find_last_of("/\\"));
  std::cout << "[" << path.substr(pos == std::string::npos ? 0 : pos + 1) << ":" << line << "] "
            << message.ToString() << std::endl;
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
