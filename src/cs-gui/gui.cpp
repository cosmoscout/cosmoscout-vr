////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gui.hpp"

#include "internal/WebApp.hpp"

#include <iostream>

namespace cs::gui {

namespace {
CefRefPtr<detail::WebApp> app;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void executeWebProcess(int argc, char* argv[]) {
  app        = new detail::WebApp(argc, argv, false);
  int result = CefExecuteProcess(app->GetArgs(), app, nullptr);
  if (result >= 0) {
    // Child proccess has endend, so exit.
    exit(result);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init() {

  if (!app) {
    std::cerr << "Failed to initialize gui: Please call "
              << "executeWebProcess() before init()!" << std::endl;
    return;
  }

  // For some reason CefInitialize changes the global locale. We therefore store
  // it here and reset it at the end of this method.
  std::locale current_locale;

  CefSettings settings;
  settings.command_line_args_disabled   = true;
  settings.no_sandbox                   = true;
  settings.remote_debugging_port        = 8999;
  settings.windowless_rendering_enabled = true;

  if (!CefInitialize(app->GetArgs(), settings, app, nullptr)) {
    std::cerr << "Failed to initialize CEF. Gui will not work at all" << std::endl;
  }

  std::locale::global(current_locale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void cleanUp() {
  CefShutdown();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void update() {
  CefDoMessageLoopWork();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
