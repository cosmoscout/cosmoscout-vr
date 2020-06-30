////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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
