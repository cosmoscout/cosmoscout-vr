////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "gui.hpp"

#include "internal/WebApp.hpp"
#include "logger.hpp"

#include <filesystem>
#include <iostream>

namespace cs::gui {

namespace {
CefRefPtr<detail::WebApp> app;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
void executeWebProcess(int argc, char* argv[]) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  app        = new detail::WebApp(argc, argv, false);
  int result = CefExecuteProcess(app->GetArgs(), app, nullptr);
  if (result >= 0) {
    // Child proccess has endend, so exit.
    exit(result);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::filesystem::path getExecutablePath() {
#ifdef __linux__
  std::filesystem::path path = "/proc/self/exe";
  return std::filesystem::read_symlink(path);
#elif _WIN32
  wchar_t buffer[MAX_PATH];
  GetModuleFileNameW(nullptr, buffer, MAX_PATH);
  return {buffer};
#else
  return {};
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init() {

  if (!app) {
    logger().error("Failed to initialize gui: Please call executeWebProcess() before init()!");
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

  std::filesystem::path exePath = getExecutablePath();
  std::filesystem::path exeDir  = exePath.parent_path();

  std::filesystem::path localesPath = exeDir / "locales";
  CefString(&settings.locales_dir_path).FromString(localesPath.string());

  std::filesystem::path cachePath = exeDir / "cef_cache";
  CefString(&settings.root_cache_path).FromString(cachePath.string());

  if (!CefInitialize(app->GetArgs(), settings, app, nullptr)) {
    logger().error("Failed to initialize CEF. Gui will not work at all.");
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
