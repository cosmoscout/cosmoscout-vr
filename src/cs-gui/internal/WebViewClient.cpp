////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebViewClient.hpp"

#include <fstream>
#include <iostream>
#include <utility>

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebViewClient::OnProcessMessageReceived(CefRefPtr<CefBrowser> browser,
    CefRefPtr<CefFrame> frame, CefProcessId source_process, CefRefPtr<CefProcessMessage> message) {

  if (message->GetName() == "call_native") {

    std::string name(message->GetArgumentList()->GetString(0).ToString());
    auto        callback(js_callbacks_.find(name));

    if (callback == js_callbacks_.end()) {
      std::cout << "Cannot call function '" << name
                << "': No callback is registered for this function name!" << std::endl;
      return true;
    }

    std::vector<std::any> args;

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
      default:
        break;
      }
    }

    callback->second(args);

    return true;
  } else if (message->GetName() == "error") {
    std::cerr << message->GetArgumentList()->GetString(0).ToString() << std::endl;
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebViewClient::RegisterJSCallback(
    std::string const& name, std::function<void(std::vector<std::any> const&)> callback) {
  js_callbacks_[name] = std::move(callback);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebViewClient::UnregisterJSCallback(std::string const& name) {
  js_callbacks_.erase(name);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
