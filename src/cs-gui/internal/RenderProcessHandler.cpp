////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "RenderProcessHandler.hpp"

#include "JSHandler.hpp"

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

void RenderProcessHandler::OnContextCreated(
    CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> /*frame*/, CefRefPtr<CefV8Context> context) {

  CefRefPtr<CefV8Value> object = context->GetGlobal();
  CefRefPtr<CefV8Value> func   = CefV8Value::CreateFunction("callNative", new JSHandler(browser));
  object->SetValue("callNative", func, V8_PROPERTY_ATTRIBUTE_NONE);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
