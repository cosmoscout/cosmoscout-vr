////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../cs-core/Settings.hpp"
#include "../cs-core/logger.hpp"
#include "../cs-graphics/logger.hpp"
#include "../cs-gui/gui.hpp"
#include "../cs-gui/logger.hpp"
#include "../cs-scene/logger.hpp"
#include "../cs-utils/CommandLine.hpp"
#include "../cs-utils/doctest.hpp"
#include "../cs-utils/logger.hpp"
#include "Application.hpp"
#include "cs-version.hpp"

#include <VistaKernel/VistaSystem.h>

#ifdef _WIN64
extern "C" {
// This tells Windows to use the dedicated NVIDIA GPU over Intel integrated graphics.
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;

// This tells Windows to use the dedicated AMD GPU over Intel integrated graphics.
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  // Launch GUI child processes. For each GUI web site, a separate process is spawned by the
  // Chromium Embedded Framework. For the main process, this method returns immediately, for all
  // others it blocks until the child process has terminated.
  cs::gui::executeWebProcess(argc, argv);

  // setup loggers ---------------------------------------------------------------------------------

  // Create default loggers. The log level will be set once the settings are read.
  spdlog::set_default_logger(cs::utils::logger::createLogger("cosmoscout-vr"));

  cs::core::logger::init();
  cs::graphics::logger::init();
  cs::gui::logger::init();
  cs::scene::logger::init();
  cs::utils::logger::init();

  // parse program options -------------------------------------------------------------------------

  // These are the default values for the options.
  std::string settingsFile   = "../share/config/simple_desktop.json";
  bool        runTests       = false;
  bool        printHelp      = false;
  bool        printVistaHelp = false;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Welcome to CosmoScout VR! Here are the available options:");
  args.addArgument({"-s", "--settings"}, &settingsFile,
      "JSON file containing settings (default: " + settingsFile + ")");
  args.addArgument({"-h", "--help"}, &printHelp, "Print this help.");
  args.addArgument({"-v", "--vistahelp"}, &printVistaHelp, "Print help for vista options.");

#ifndef DOCTEST_CONFIG_DISABLE
  args.addArgument({"-t", "--run-tests"}, &runTests, "Runs all unit tests.");
#endif

  // Then do the actual parsing.
  try {
    args.parse(argc, argv);
  } catch (std::runtime_error const& e) {
    spdlog::error("Failed to parse command line arguments: {}", e.what());
    return 1;
  }

  // Run all registered tests.
  if (runTests) {
    Application::testLoadAllPlugins();
    doctest::Context context(argc, argv);
    return context.run();
  }

  // When printHelp was set to true, we print a help message and exit.
  if (printHelp) {
    args.printHelp();
    return 0;
  }

  // When printVistaHelp was set to true, we print a help message and exit.
  if (printVistaHelp) {
    VistaSystem::ArgHelpMsg(argv[0], &std::cout);
    return 0;
  }

  // read settings ---------------------------------------------------------------------------------

  auto settings = std::make_shared<cs::core::Settings>();
  try {
    settings->read(settingsFile);
  } catch (std::exception& e) {
    spdlog::error("Failed to read settings: {}", e.what());
    return 1;
  }

  // configure loggers -----------------------------------------------------------------------------

  // Once we have read the settings, we can set the log level.
  settings->mLogLevel.mConsole.connectAndTouch(
      [](auto level) { cs::utils::logger::setCoutLogLevel(level); });
  settings->mLogLevel.mFile.connectAndTouch(
      [](auto level) { cs::utils::logger::setFileLogLevel(level); });
  settings->mLogLevel.mScreen.connectAndTouch(
      [](auto level) { cs::utils::logger::setSignalLogLevel(level); });

  // Print a nifty welcome message!
  spdlog::info("Welcome to CosmoScout VR v" + CS_PROJECT_VERSION + "!");

  // start application -----------------------------------------------------------------------------

  try {
    // First we need a VistaSystem.
    auto pVistaSystem = new VistaSystem();

    // ViSTA is configured with plenty of ini files. The ini files of CosmoScout VR reside in a
    // specific directory, so we have to add this directory to the search paths.
    pVistaSystem->SetIniSearchPaths({"../share/config/vista"});

    // The Application contains a lot of initialization code and the frame update.
    Application app(settings);
    pVistaSystem->SetFrameLoop(&app, true);

    // Now run the program!
    if (pVistaSystem->Init(argc, argv)) {
      pVistaSystem->Run();
    }

  } catch (VistaExceptionBase& e) {
    spdlog::error("Caught unexpected VistaException: {}", e.what());
    return 1;
  } catch (std::exception& e) {
    spdlog::error("Caught unexpected std::exception: {}", e.what());
    return 1;
  }

  spdlog::info("Shutdown complete. Fare well!");

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
