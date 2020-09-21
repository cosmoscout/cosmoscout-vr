////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_DETAIL_WEBVIEWCLIENT_HPP
#define CS_GUI_DETAIL_WEBVIEWCLIENT_HPP

#include "../types.hpp"

#include "DisplayHandler.hpp"
#include "LifeSpanHandler.hpp"
#include "LoadHandler.hpp"
#include "RenderHandler.hpp"
#include "RequestHandler.hpp"
#include "ResourceRequestHandler.hpp"

#include <include/cef_client.h>
#include <include/cef_render_handler.h>
#include <include/cef_v8.h>
#include <include/wrapper/cef_stream_resource_handler.h>
#include <optional>
#include <unordered_map>

namespace cs::gui::detail {

/// This interface implementation provides cef with out custom handlers.
class WebViewClient : public CefClient {

 public:
  WebViewClient() = default;
  ~WebViewClient() override;

  WebViewClient(WebViewClient const& other) = delete;
  WebViewClient(WebViewClient&& other)      = delete;

  WebViewClient& operator=(WebViewClient const& other) = delete;
  WebViewClient& operator=(WebViewClient&& other) = delete;

  /// Registers callback functions for Javascript. Registering the same name twice will override the
  /// first callback.
  void RegisterJSCallback(
      std::string const& name, std::function<void(std::vector<std::optional<JSType>>&&)> callback);

  /// Unregisters a JavaScript callback.
  void UnregisterJSCallback(std::string const& name);

  /// Returns the concrete implementation of the LifeSpanHandler.
  CefRefPtr<LifeSpanHandler> GetInternalLifeSpanHandler() {
    return mLifeSpanHandler;
  }

  /// Returns the concrete implementation of the DisplayHandler.
  CefRefPtr<DisplayHandler> GetInternalDisplayHandler() {
    return mDisplayHandler;
  }

  /// Returns the concrete implementation of the RenderHandler.
  CefRefPtr<RenderHandler> GetInternalRenderHandler() {
    return mRenderHandler;
  }

  /// Returns the concrete implementation of the LoadHandler.
  CefRefPtr<LoadHandler> GetInternalLoadHandler() {
    return mLoadHandler;
  }

  /// Returns the concrete implementation of the RequestHandler.
  CefRefPtr<RequestHandler> GetInternalRequestHandler() {
    return mRequestHandler;
  }

  // CefClient interface ---------------------------------------------------------------------------

  CefRefPtr<CefLifeSpanHandler> GetLifeSpanHandler() override {
    return mLifeSpanHandler;
  }
  CefRefPtr<CefDisplayHandler> GetDisplayHandler() override {
    return mDisplayHandler;
  }
  CefRefPtr<CefRenderHandler> GetRenderHandler() override {
    return mRenderHandler;
  }
  CefRefPtr<CefLoadHandler> GetLoadHandler() override {
    return mLoadHandler;
  }
  CefRefPtr<CefRequestHandler> GetRequestHandler() override {
    return mRequestHandler;
  }

  /// Forwards JS callbacks for execution.
  bool OnProcessMessageReceived(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
      CefProcessId source_process, CefRefPtr<CefProcessMessage> message) override;

 private:
  IMPLEMENT_REFCOUNTING(WebViewClient);

  CefRefPtr<LifeSpanHandler> mLifeSpanHandler = new LifeSpanHandler();
  CefRefPtr<DisplayHandler>  mDisplayHandler  = new DisplayHandler();
  CefRefPtr<RenderHandler>   mRenderHandler   = new RenderHandler();
  CefRefPtr<LoadHandler>     mLoadHandler     = new LoadHandler();
  CefRefPtr<RequestHandler>  mRequestHandler  = new RequestHandler();

  std::unordered_map<std::string, std::function<void(std::vector<std::optional<JSType>>&&)>>
      mJSCallbacks;
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_WEBVIEWCLIENT_HPP
