////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebView.hpp"

#include "internal/WebViewClient.hpp"

#include <include/cef_app.h>
#include <thread>

namespace cs::gui {

////////////////////////////////////////////////////////////////////////////////////////////////////

class DevToolsClient : public CefClient {
 public:
  IMPLEMENT_REFCOUNTING(DevToolsClient);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

WebView::WebView(const std::string& url, int width, int height, bool allowLocalFileAccess)
    : mClient(new detail::WebViewClient())
    , mInteractive(true)
    , mMouseX(0)
    , mMouseY(0)
    , mMouseModifiers(0) {
  resize(width, height);

  CefWindowInfo info;
  info.width  = width;
  info.height = height;
  info.SetAsWindowless(0);

  CefBrowserSettings browserSettings;
  browserSettings.windowless_frame_rate = 60;
  browserSettings.web_security          = allowLocalFileAccess ? STATE_DISABLED : STATE_ENABLED;

  mBrowser =
      CefBrowserHost::CreateBrowserSync(info, mClient, url, browserSettings, nullptr, nullptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebView::~WebView() {
  auto host = mBrowser->GetHost();
  while (!host->TryCloseBrowser()) {
    CefDoMessageLoopWork();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // magically called by the code above
  // delete mClient;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::setDrawCallback(DrawCallback const& callback) {
  mClient->GetInternalRenderHandler()->SetDrawCallback(callback);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::setCursorChangeCallback(CursorChangeCallback const& callback) {
  mClient->GetInternalRenderHandler()->SetCursorChangeCallback(callback);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::resize(int width, int height) const {
  mClient->GetInternalRenderHandler()->Resize(width, height);

  if (mBrowser) {
    mBrowser->GetHost()->WasResized();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebView::getColor(int x, int y, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) const {
  return mClient->GetInternalRenderHandler()->GetColor(x, y, r, g, b, a);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint8_t WebView::getRed(int x, int y) const {
  uint8_t value(0), tmp(0);
  mClient->GetInternalRenderHandler()->GetColor(x, y, value, tmp, tmp, tmp);
  return value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint8_t WebView::getGreen(int x, int y) const {
  uint8_t value(0), tmp(0);
  mClient->GetInternalRenderHandler()->GetColor(x, y, tmp, value, tmp, tmp);
  return value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint8_t WebView::getBlue(int x, int y) const {
  uint8_t value(0), tmp(0);
  mClient->GetInternalRenderHandler()->GetColor(x, y, tmp, tmp, value, tmp);
  return value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint8_t WebView::getAlpha(int x, int y) const {
  uint8_t value(0), tmp(0);
  mClient->GetInternalRenderHandler()->GetColor(x, y, tmp, tmp, tmp, value);
  return value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebView::getIsInteractive() const {
  return mInteractive;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::setIsInteractive(bool interactive) {
  mInteractive = interactive;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int WebView::getWidth() const {
  return mClient->GetInternalRenderHandler()->GetWidth();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int WebView::getHeight() const {
  return mClient->GetInternalRenderHandler()->GetHeight();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::waitForFinishedLoading() const {
  mClient->GetInternalLoadHandler()->WaitForFinishedLoading();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::reload(bool ignoreCache) const {
  if (ignoreCache)
    mBrowser->ReloadIgnoreCache();
  else
    mBrowser->Reload();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::goBack() const {
  mBrowser->GoBack();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::goForward() const {
  mBrowser->GoForward();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::cut() const {
  mBrowser->GetFocusedFrame()->Cut();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::copy() const {
  mBrowser->GetFocusedFrame()->Copy();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::paste() const {
  mBrowser->GetFocusedFrame()->Paste();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::remove() const {
  mBrowser->GetFocusedFrame()->Delete();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::undo() const {
  mBrowser->GetFocusedFrame()->Undo();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::redo() const {
  mBrowser->GetFocusedFrame()->Redo();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::injectFocusEvent(bool focus) {
  if (!mInteractive)
    return;
  mBrowser->GetHost()->SendFocusEvent(focus);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::injectMouseEvent(MouseEvent const& event) {
  if (!mInteractive)
    return;

  CefMouseEvent cef_event;
  cef_event.modifiers = (uint32)mMouseModifiers;
  cef_event.x         = mMouseX;
  cef_event.y         = mMouseY;

  switch (event.mType) {
  case MouseEvent::Type::eMove:
    cef_event.x = mMouseX = event.mX;
    cef_event.y = mMouseY = event.mY;
    mBrowser->GetHost()->SendMouseMoveEvent(cef_event, false);
    break;

  case MouseEvent::Type::eLeave:
    mBrowser->GetHost()->SendMouseMoveEvent(cef_event, true);
    break;

  case MouseEvent::Type::eScroll:
    mBrowser->GetHost()->SendMouseWheelEvent(cef_event, event.mX, event.mY);
    break;

  case MouseEvent::Type::ePress:
    if (event.mButton == Button::eLeft) {
      mMouseModifiers |= int(Modifier::eLeftButton);
      double elpasedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - mLastClick)
                               .count();
      if (elpasedTime < 200) {
        mClickCount++;
      } else {
        mClickCount = 1;
      }
      mLastClick = std::chrono::steady_clock::now();
    } else if (event.mButton == Button::eRight) {
      mMouseModifiers |= int(Modifier::eRightButton);
    } else if (event.mButton == Button::eMiddle) {
      mMouseModifiers |= int(Modifier::eMiddleButton);
    }

    cef_event.modifiers = (uint32)mMouseModifiers;
    mBrowser->GetHost()->SendMouseClickEvent(
        cef_event, (cef_mouse_button_type_t)event.mButton, false, mClickCount);
    break;

  case MouseEvent::Type::eRelease:
    if (event.mButton == Button::eLeft) {
      mMouseModifiers &= ~int(Modifier::eLeftButton);
    } else if (event.mButton == Button::eRight) {
      mMouseModifiers &= ~int(Modifier::eRightButton);
    } else if (event.mButton == Button::eMiddle) {
      mMouseModifiers &= ~int(Modifier::eMiddleButton);
    }

    cef_event.modifiers = (uint32)mMouseModifiers;
    mBrowser->GetHost()->SendMouseClickEvent(
        cef_event, (cef_mouse_button_type_t)event.mButton, true, mClickCount);
    break;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::injectKeyEvent(KeyEvent const& event) {
  if (!mInteractive || event.mType == KeyEvent::Type::eInvalid) {
    return;
  }

  CefKeyEvent cef_event;
  cef_event.modifiers               = event.mModifiers;
  cef_event.character               = event.mCharacter;
  cef_event.is_system_key           = false;
  cef_event.windows_key_code        = (int)event.mKey;
  cef_event.focus_on_editable_field = true;

  if (event.mType == KeyEvent::Type::ePress) {
    cef_event.type = KEYEVENT_KEYDOWN;
  } else if (event.mType == KeyEvent::Type::eRelease) {
    cef_event.type = KEYEVENT_KEYUP;
  } else {
    cef_event.type = KEYEVENT_CHAR;
  }

  mBrowser->GetHost()->SendKeyEvent(cef_event);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::callJavascriptImpl(
    std::string const& function, std::vector<std::string> const& args) const {
  std::string call(function + "( ");
  for (auto& s : args) {
    call += s + ",";
  }
  call.back() = ')';

  CefRefPtr<CefFrame> frame = mBrowser->GetMainFrame();
  frame->ExecuteJavaScript(call, frame->GetURL(), 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::executeJavascript(std::string const& code) const {
  CefRefPtr<CefFrame> frame = mBrowser->GetMainFrame();
  frame->ExecuteJavaScript(code, frame->GetURL(), 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::unregisterCallback(std::string const& name) {
  mClient->UnregisterJSCallback(name);

  std::string cmd = R"(
    if (typeof CosmoScout !== 'undefined') {
      delete CosmoScout.callbacks.$;
    }
  )";

  utils::replaceString(cmd, "$", name);
  executeJavascript(cmd);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebView::registerJSCallbackImpl(
    std::string const& name, std::function<void(std::vector<std::any> const&)> const& callback) {
  mClient->RegisterJSCallback(name, callback);

  std::string cmd = R"(
    if (typeof CosmoScout !== 'undefined') {
      CosmoScout.callbacks.$ = (...args) => window.call_native('$', ...args);
    }
  )";

  utils::replaceString(cmd, "$", name);
  executeJavascript(cmd);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
