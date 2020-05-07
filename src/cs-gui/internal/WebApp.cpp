////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebApp.hpp"

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

// NOLINTNEXTLINE
WebApp::WebApp(int argc, char* argv[], bool hardware_accelerated)
#ifdef _WIN32
    : mArgs(GetModuleHandleW(nullptr))
#else
    : mArgs(argc, argv)
#endif
    , mHardwareAccelerated(hardware_accelerated) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CefMainArgs const& WebApp::GetArgs() {
  return mArgs;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebApp::OnBeforeCommandLineProcessing(
    const CefString& process_type, CefRefPtr<CefCommandLine> command_line) {

  if (process_type.empty()) {
    command_line->AppendSwitch("enable-overlay-scrollbar");
    command_line->AppendSwitch("enable-begin-frame-scheduling");

    if (!mHardwareAccelerated) {
      command_line->AppendSwitch("disable-gpu");
      command_line->AppendSwitch("disable-gpu-compositing");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
