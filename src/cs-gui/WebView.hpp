////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_WEBVIEW_HPP
#define CS_GUI_WEBVIEW_HPP

#include "../cs-utils/utils.hpp"
#include "KeyEvent.hpp"
#include "MouseEvent.hpp"

#include <any>
#include <chrono>
#include <include/cef_client.h>
#include <iostream>
#include <spdlog/spdlog.h>

namespace cs::gui {

/// A WebView is wrapper of an HTML page. It allows for registering C++ callbacks which can be
/// called from JavaScript and allows executing JavaScript code in the context of the website from
/// C++. Usually you will not instantiate this class directly, you will rather use the GuiItem
/// class.
/// For debugging, you can use Chromium's developper tools. Once the applications is running, you
/// can navigate to http://127.0.0.1:8999/ with your Chromium based browser in order to inspect the
/// individual WebViews of CosmoScout VR.
class CS_GUI_EXPORT WebView {
 public:
  /// Creates a new WebView for the given page at the location of the URL.
  WebView(const std::string& url, int width, int height, bool allowLocalFileAccess = false);
  virtual ~WebView();

  /// Registers a callback that is called, when the page is redrawn.
  void setDrawCallback(DrawCallback const& callback);

  /// Registers a callback, that is called, when the cursor changes its appearance.
  void setCursorChangeCallback(CursorChangeCallback const& callback);

  /// Calls an existing Javascript function. You can pass as many arguments as you like. They will
  /// be converted to std::strings, so on the JavaScript side you will have to convert them back.
  ///
  /// @param function The name of the function.
  /// @param a        The arguments of the function. Each arguments type must be convertible to a
  ///                 string be either providing a definition for core::utils::toString or by
  ///                 implementing the operator<<() for that type.
  template <typename... Args>
  void callJavascript(std::string const& function, Args&&... a) const {
    std::vector<std::string> args = {(utils::toString(a))...};
    callJavascriptImpl(function, args);
  }

  /// Execute Javascript code.
  void executeJavascript(std::string const& code) const;

  /// Register a callback which can be called from Javascript with the
  /// "window.call_native('callback_name', ... args ...)" function. Registering the same name twice
  /// will override the first callback.
  /// This first version takes no arguments from the JavaScript side. There are other versions
  /// below, which take up to six arguments. JavaScript variables passed to the window.call_native
  /// function will be converted to C++ types. This works for integer, doubles, booleans and
  /// std::strings.
  ///
  /// @param name     Name of the callback.
  /// @param callback The function to execute when the HTML-Element fires a change event.
  void registerCallback(std::string const& name, std::function<void()> const& callback) {
    registerJSCallbackImpl(
        name, [this, callback](std::vector<std::any> const& args) { callback(); });
  }

  /// See documentation above.
  template <typename A>
  void registerCallback(std::string const& name, std::function<void(A)> const& callback) {
    registerJSCallbackImpl(name, [this, name, callback](std::vector<std::any> const& args) {
      try {
        callback(std::any_cast<A>(args[0]));
      } catch (std::bad_any_cast const& e) {
        spdlog::error("Cannot execute javascript call '{}': {}", name, e.what());
      }
    });
  }

  /// See documentation above.
  template <typename A, typename B>
  void registerCallback(std::string const& name, std::function<void(A, B)> const& callback) {
    registerJSCallbackImpl(name, [this, name, callback](std::vector<std::any> const& args) {
      try {
        callback(std::any_cast<A>(args[0]), std::any_cast<B>(args[1]));
      } catch (std::bad_any_cast const& e) {
        spdlog::error("Cannot execute javascript call '{}': {}", name, e.what());
      }
    });
  }

  /// See documentation above.
  template <typename A, typename B, typename C>
  void registerCallback(std::string const& name, std::function<void(A, B, C)> const& callback) {
    registerJSCallbackImpl(name, [this, name, callback](std::vector<std::any> const& args) {
      try {
        callback(std::any_cast<A>(args[0]), std::any_cast<B>(args[1]), std::any_cast<C>(args[2]));
      } catch (std::bad_any_cast const& e) {
        spdlog::error("Cannot execute javascript call '{}': {}", name, e.what());
      }
    });
  }

  /// See documentation above.
  template <typename A, typename B, typename C, typename D>
  void registerCallback(std::string const& name, std::function<void(A, B, C, D)> const& callback) {
    registerJSCallbackImpl(name, [this, name, callback](std::vector<std::any> const& args) {
      try {
        callback(std::any_cast<A>(args[0]), std::any_cast<B>(args[1]), std::any_cast<C>(args[2]),
            std::any_cast<D>(args[3]));
      } catch (std::bad_any_cast const& e) {
        spdlog::error("Cannot execute javascript call '{}': {}", name, e.what());
      }
    });
  }

  /// See documentation above.
  template <typename A, typename B, typename C, typename D, typename E>
  void registerCallback(
      std::string const& name, std::function<void(A, B, C, D, E)> const& callback) {
    registerJSCallbackImpl(name, [this, name, callback](std::vector<std::any> const& args) {
      try {
        callback(std::any_cast<A>(args[0]), std::any_cast<B>(args[1]), std::any_cast<C>(args[2]),
            std::any_cast<D>(args[3]), std::any_cast<E>(args[4]));
      } catch (std::bad_any_cast const& e) {
        spdlog::error("Cannot execute javascript call '{}': {}", name, e.what());
      }
    });
  }

  /// See documentation above.
  template <typename A, typename B, typename C, typename D, typename E, typename F>
  void registerCallback(
      std::string const& name, std::function<void(A, B, C, D, E, F)> const& callback) {
    registerJSCallbackImpl(name, [this, name, callback](std::vector<std::any> const& args) {
      try {
        callback(std::any_cast<A>(args[0]), std::any_cast<B>(args[1]), std::any_cast<C>(args[2]),
            std::any_cast<D>(args[3]), std::any_cast<E>(args[4]), std::any_cast<F>(args[5]));
      } catch (std::bad_any_cast const& e) {
        spdlog::error("Cannot execute javascript call '{}': {}", name, e.what());
      }
    });
  }

  /// Unregisters a JavaScript callback.
  void unregisterCallback(std::string const& name);

  /// Resize the pages contents to match the given width and height.
  virtual void resize(int width, int height) const;

  /// Gives back the color information for the given pixel.
  ///
  /// @return false if the information is not available.
  virtual bool    getColor(int x, int y, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) const;
  virtual uint8_t getRed(int x, int y) const;   ///< Gives the red value at the given coordinates.
  virtual uint8_t getGreen(int x, int y) const; ///< Gives the green value at the given coordinates.
  virtual uint8_t getBlue(int x, int y) const;  ///< Gives the blue value at the given coordinates.
  virtual uint8_t getAlpha(int x, int y) const; ///< Gives the alpha value at the given coordinates.

  /// The interactive state determines if a user can interact with the HTML contents. If set to
  /// false all inputs will be ignored. This might increase performance.
  virtual bool getIsInteractive() const;
  virtual void setIsInteractive(bool interactive);

  /// Returns the current size of the web page.
  virtual int getWidth() const;
  virtual int getHeight() const;

  /// Waits for the page to load properly. This function should be called, before displaying the
  /// page.
  virtual void waitForFinishedLoading() const;

  /// Reloads the page.
  ///
  /// @param ignoreCache If set to true the site will not be using cached data.
  virtual void reload(bool ignoreCache = false) const;

  /// When the user clicked on a hyperlink on the web page, this functionality can be used to move
  /// forward or backwards in the history.
  virtual void goBack() const;
  virtual void goForward() const;

  /// These could be executed when the according hot keys are pressed. The system's clipboard will
  /// be used. So the user can actually copy-paste from one WebView to another or even from an to
  /// third-party applications.
  virtual void cut() const;
  virtual void copy() const;
  virtual void paste() const;
  virtual void remove() const;
  virtual void undo() const;
  virtual void redo() const;

  /// Toggle the focused state for this page.
  virtual void injectFocusEvent(bool focus);

  /// Forward a MouseEvent to the page.
  virtual void injectMouseEvent(MouseEvent const& event);

  /// Forward a KeyEvent to the page.
  virtual void injectKeyEvent(KeyEvent const& event);

  /// These are not yet working properly. However, you can navigate to http://127.0.0.1:8999/ with
  /// your Chromium based browser in order to inspect the individual WebViews of CosmoScout VR.
  void toggleDevTools();
  void showDevTools();
  void closeDevTools();

 private:
  void callJavascriptImpl(std::string const& function, std::vector<std::string> const& args) const;
  void registerJSCallbackImpl(
      std::string const& name, std::function<void(std::vector<std::any> const&)> const& callback);

  detail::WebViewClient* mClient;
  CefRefPtr<CefBrowser>  mBrowser;

  bool mInteractive;

  // Input state.
  int mMouseX;
  int mMouseY;
  int mMouseModifiers;

  // Time point for the last left mouse click
  std::chrono::steady_clock::time_point mLastClick;

  // Count number of left mouse button clicks
  int mClickCount = 1;
};

} // namespace cs::gui

#endif // CS_GUI_WEBVIEW_HPP
