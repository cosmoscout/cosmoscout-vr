////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../cs-core/Settings.hpp"
#include "../cs-gui/gui.hpp"
#include "../cs-utils/CommandLine.hpp"
#include "Application.hpp"

#include <VistaKernel/VistaSystem.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  // Launch GUI child processes. For each GUI web site, a separate process is spawned by the
  // Chromium Embedded Framework. For the main process, this method returns immediately, for all
  // others it blocks until the child process has terminated.
  cs::gui::executeWebProcess(argc, argv);

  // parse program options -------------------------------------------------------------------------

  // These are the default values for the options.
  std::string settingsFile   = "../share/config/simple_desktop.json";
  bool        printHelp      = false;
  bool        printVistaHelp = false;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Welcome to CosmoScout VR! Here are the available options:");
  args.addArgument({"-s", "--settings"}, &settingsFile,
      "JSON file containing settings (default: " + settingsFile + ")");
  args.addArgument({"-h", "--help"}, &printHelp, "Print this help.");
  args.addArgument({"-v", "--vistahelp"}, &printVistaHelp, "Print help for vista options.");

  // Then do the actual parsing.
  try {
    args.parse(argc, argv);
  } catch (std::runtime_error const& e) {
    std::cout << e.what() << std::endl;
    return 1;
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

  cs::core::Settings settings;
  try {
    settings = cs::core::Settings::read(settingsFile);
  } catch (std::exception& e) {
    std::cerr << "Failed to read settings: " << e.what() << std::endl;
    return 1;
  }

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
    e.PrintException();
    return 1;
  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
