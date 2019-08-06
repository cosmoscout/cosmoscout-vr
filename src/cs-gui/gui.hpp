////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_GUI_HPP
#define CS_GUI_GUI_HPP

#include "cs_gui_export.hpp"

/// This namespace contains all functionality to interact with the user interface. The UI is a
/// web application running in the Chromium Embedded Framework (CEF).
namespace cs::gui {

/// Launches the UI process with the given arguments in which the Chromium Embedded Framework runs.
/// This needs to be called before init().
// TODO the name of the function is not very descriptive given its very specialized functionality.
//  A better name could be executeUIProcess() or executeCEFProcess().
CS_GUI_EXPORT void executeChildProcess(int argc, char* argv[]);

/// Initializes CEF. Needs to be called after executeChildProcess().
CS_GUI_EXPORT void init();

/// Shuts down CEF.
CS_GUI_EXPORT void cleanUp();

/// Triggers the CEF update function.
CS_GUI_EXPORT void update();

} // namespace cs::gui

#endif // CS_GUI_GUI_HPP
