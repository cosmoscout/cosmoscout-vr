////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebViewClient.hpp"

#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <utility>

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebViewClient::~WebViewClient() {
  if (mJSCallbacks.size() > 0) {
    spdlog::warn(
        "While destructing a WebViewClient there were still JavaScript callbacks registered:");

    for (auto&& i : mJSCallbacks) {
      spdlog::warn(" - {}", i.first);
      i.second = [](std::vector<std::optional<JSType>> const&) {};
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebViewClient::OnProcessMessageReceived(CefRefPtr<CefBrowser> browser,
    CefRefPtr<CefFrame> frame, CefProcessId source_process, CefRefPtr<CefProcessMessage> message) {

  if (message->GetName() == "callNative") {

    std::string name(message->GetArgumentList()->GetString(0).ToString());
    auto        callback(mJSCallbacks.find(name));

    if (callback == mJSCallbacks.end()) {
      spdlog::warn(
          "Cannot call function '{}': No callback is registered for this function name!", name);
      return true;
    }

    std::vector<std::optional<JSType>> args;

    for (int i(1); i < message->GetArgumentList()->GetSize(); ++i) {
      CefValueType type(message->GetArgumentList()->GetType((size_t)i));
      switch (type) {
      case VTYPE_DOUBLE:
        args.emplace_back(message->GetArgumentList()->GetDouble((size_t)i));
        break;
      case VTYPE_BOOL:
        args.emplace_back(message->GetArgumentList()->GetBool((size_t)i));
        break;
      case VTYPE_STRING:
        args.emplace_back(message->GetArgumentList()->GetString((size_t)i).ToString());
        break;
      case VTYPE_INVALID:
      case VTYPE_NULL:
        args.emplace_back(std::nullopt);
        break;
      default:
        spdlog::warn("Failed to parse argument {} of callback '{}': Unsupported type!", i, name);
        break;
      }
    }

    callback->second(std::move(args));

    return true;
  } else if (message->GetName() == "error") {
    spdlog::error(message->GetArgumentList()->GetString(0).ToString());
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebViewClient::RegisterJSCallback(
    std::string const& name, std::function<void(std::vector<std::optional<JSType>>&&)> callback) {
  mJSCallbacks[name] = std::move(callback);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebViewClient::UnregisterJSCallback(std::string const& name) {
  mJSCallbacks.erase(name);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
