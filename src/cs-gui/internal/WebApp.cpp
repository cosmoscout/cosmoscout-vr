////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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
    command_line->AppendSwitch("disable-web-security");
  }

#ifdef _WIN32
  if (process_type == "gpu-process") {
    command_line->AppendSwitch("disable-vulkan");
    command_line->AppendSwitchWithValue("disable-features", "Vulkan");
    command_line->AppendSwitchWithValue("use-angle", "d3d11");

    if (!mHardwareAccelerated) {
      command_line->AppendSwitch("disable-gpu");
      command_line->AppendSwitch("disable-gpu-compositing");
    }
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
