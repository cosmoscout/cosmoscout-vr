////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "JSHandler.hpp"

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

bool JSHandler::Execute(const CefString& name, CefRefPtr<CefV8Value> /*object*/,
    const CefV8ValueList& arguments, CefRefPtr<CefV8Value>& /*retval*/, CefString& /*exception*/) {

  if (name != "callNative") {
    SendError("Unknown Javascript function name!");
    return false;
  }

  if (arguments.empty()) {
    SendError("window.callNative function requires at least one argument!");
    return false;
  }

  CefRefPtr<CefProcessMessage> msg = CefProcessMessage::Create(name);
  msg->GetArgumentList()->SetString(0, arguments[0]->GetStringValue());

  bool success(true);

  for (size_t i(1); i < arguments.size(); ++i) {
    if (arguments[i]->IsDouble()) {
      msg->GetArgumentList()->SetDouble(i, arguments[i]->GetDoubleValue());
    } else if (arguments[i]->IsBool()) {
      msg->GetArgumentList()->SetBool(i, arguments[i]->GetBoolValue());
    } else if (arguments[i]->IsString()) {
      msg->GetArgumentList()->SetString(i, arguments[i]->GetStringValue());
    } else if (arguments[i]->IsNull() || arguments[i]->IsUndefined()) {
      msg->GetArgumentList()->SetNull(i);
    } else {
      std::stringstream sstr;
      sstr << "Failed to handle window.callNative call. Argument " << i
           << " has an unsupported type. Only Double, Bool and String are "
           << "supported.";
      SendError(sstr.str());
      success = false;
    }
  }

  if (success) {
    mBrowser->GetMainFrame()->SendProcessMessage(PID_BROWSER, msg);
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void JSHandler::SendError(std::string const& message) const {
  CefRefPtr<CefProcessMessage> msg = CefProcessMessage::Create("error");
  msg->GetArgumentList()->SetString(0, message);
  mBrowser->GetMainFrame()->SendProcessMessage(PID_BROWSER, msg);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
