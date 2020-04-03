////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_GUI_HPP
#define CS_GUI_GUI_HPP

#include "cs_gui_export.hpp"

/// This namespace contains global functionality to interact with the user interface. The UI is a
/// web application running in the Chromium Embedded Framework (CEF).
namespace cs::gui {

/// Launches GUI child processes. For each GUI web site, a separate process is spawned by the
/// Chromium Embedded Framework. For the main process, this method returns immediately, for all
/// others it blocks until the child process has terminated.
CS_GUI_EXPORT void executeWebProcess(int argc, char* argv[]); // NOLINT(modernize-avoid-c-arrays)

/// Initializes CEF. Needs to be called after executeWebProcess().
CS_GUI_EXPORT void init();

/// Shuts down CEF.
CS_GUI_EXPORT void cleanUp();

/// Triggers the CEF update function. This should be called once a frame.
CS_GUI_EXPORT void update();

} // namespace cs::gui

#endif // CS_GUI_GUI_HPP
