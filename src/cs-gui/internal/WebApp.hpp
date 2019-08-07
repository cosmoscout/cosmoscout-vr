////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_DETAIL_WEB_APP_HPP
#define CS_GUI_DETAIL_WEB_APP_HPP

#include "RenderProcessHandler.hpp"

#include <include/cef_app.h>

namespace cs::gui::detail {

/// Implements the CefApp interface.
class WebApp : public CefApp {

 public:
  /// If hardware_accelerated is set to true, webgl will be available, but the
  /// overall performance is likely to be worse.
  WebApp(int argc, char* argv[], bool hardware_accelerated);

  CefMainArgs const& GetArgs();

  /// Returns this.
  CefRefPtr<CefRenderProcessHandler> GetRenderProcessHandler() override {
    return mRenderProcessHandler;
  }

  /// This is called just before the command line arguments are parsed by CEF. We
  /// use this opportunity to add some configuration parameters.
  void OnBeforeCommandLineProcessing(
      CefString const& process_type, CefRefPtr<CefCommandLine> command_line) override;

 private:
  IMPLEMENT_REFCOUNTING(WebApp);

  CefRefPtr<RenderProcessHandler> mRenderProcessHandler = new RenderProcessHandler();

  CefMainArgs mArgs;
  bool        mHardwareAccelerated;
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_WEB_APP_HPP
