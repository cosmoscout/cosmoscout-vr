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
#include "logger.hpp"

#include <chrono>
#include <include/cef_client.h>
#include <iostream>
#include <optional>
#include <typeindex>

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

  WebView(WebView const& other) = delete;
  WebView(WebView&& other)      = delete;

  WebView& operator=(WebView const& other) = delete;
  WebView& operator=(WebView&& other) = delete;

  virtual ~WebView();

  /// Registers a callback that is called, when the page is redrawn.
  void setDrawCallback(DrawCallback const& callback);

  /// The given callback is fired when the cursor icon should change.
  void setCursorChangeCallback(CursorChangeCallback const& callback);

  /// The given callback is fired when the active gui element wants to receive keyboard events.
  void setRequestKeyboardFocusCallback(RequestKeyboardFocusCallback const& callback);

  /// Calls an existing Javascript function. You can pass as many arguments as you like. They will
  /// be converted to std::strings, so on the JavaScript side you will have to convert them back.
  ///
  /// @param function The name of the function.
  /// @param a        The arguments of the function. Each arguments type must be convertible to a
  ///                 string be either providing a definition for core::utils::toString or by
  ///                 implementing the operator<<() for that type.
  template <typename... Args>
  void callJavascript(std::string const& function, Args&&... a) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
    std::vector<std::string> args = {(utils::toString(a))...};
    callJavascriptImpl(function, args);
  }

  /// Execute Javascript code.
  void executeJavascript(std::string const& code) const;

  /// Register a callback which can be called from Javascript with the
  /// "window.callNative('callback_name', ... args ...)" function. Callbacks are also registered as
  /// CosmoScout.callbacks.callback_name(... args ...). For the latter to work, the WebView has to
  /// have finished loading. So please call waitForFinishedLoading() before calling these methods.
  /// It is fine (and encouraged) to have dots in the callback name in order to create scopes.
  /// Registering the same name twice will override the first callback. The first version of
  /// registerCallback() takes no arguments from the JavaScript side. There is another other
  /// versions below, which takes arbitrary arguments. JavaScript variables passed to the
  /// window.callNative function will be converted to C++ types. This works for doubles, bools and
  /// std::strings.
  ///
  /// @param name     Name of the callback.
  /// @param comment  The comment will be visible when inspecting the CosmoScout.callbacks object
  ///                 in a interactive console.
  /// @param callback The function to execute when the HTML-Element fires a change event.
  void registerCallback(
      std::string const& name, std::string const& comment, std::function<void()> const& callback);

  /// See documentation above.
  template <typename... Args>
  void registerCallback(std::string const& name, std::string const& comment,
      std::function<void(Args...)> const& callback) {
    assertJavaScriptTypes<Args...>();
    registerCallbackWrapper(name, comment, callback, std::index_sequence_for<Args...>{});
  }

  /// Unregisters a JavaScript callback.
  void unregisterCallback(std::string const& name);

  /// Resize the pages contents to match the given width and height.
  virtual void resize(int width, int height) const;

  /// Sets a scale / zoom multiplicator. Should be greater than zero; one is the default level.
  /// You should waitForFinishedLoading() before calling this method.
  ///
  /// Due to a 'feature' of the Chromium Embedded Framework, all WebViews sharing the same host and
  /// scheme are zoomed simultaneously. See the below link for a discussion:
  /// www.magpcss.org/ceforum/viewtopic.php?f=6&t=15167&sid=a17633152e4f453b651d898231f0e76d
  ///
  /// Because all WebViews using a "file://../..." URL effectively share the same 'host' and scheme,
  /// all will be affected by a call to this method. As a workaround, our
  /// internal/ResourceRequestHandler.cpp will ignore any leading "{...}" in paths when loading
  /// local files. This way, if you want to load the files "file://../gui/a.html" and
  /// "file://../gui/b.html", you could load them for example with "file://{zoomA}../gui/a.html" and
  /// "file://{zoomB}../gui/b.html" in order to prevent simultaneous zooming.
  virtual void setZoomFactor(double factor) const;

  /// Gives back the color information for the given pixel.
  ///
  /// @return false if the information is not available.
  virtual bool    getColor(int x, int y, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) const;
  virtual uint8_t getRed(int x, int y) const;   ///< Gives the red value at the given coordinates.
  virtual uint8_t getGreen(int x, int y) const; ///< Gives the green value at the given coordinates.
  virtual uint8_t getBlue(int x, int y) const;  ///< Gives the blue value at the given coordinates.
  virtual uint8_t getAlpha(int x, int y) const; ///< Gives the alpha value at the given coordinates.

  /// The interactive state determines if a user can interact with the HTML contents. If set to
  /// false all inputs will be ignored. This might increase performance. Default is true.
  virtual bool getIsInteractive() const;
  virtual void setIsInteractive(bool interactive);

  /// If set to false, all mouse scroll events will be ignored. This may increase performance but
  /// more importantly, things like zooming will work even if the mouse is hovering this webview.
  virtual bool getCanScroll() const;
  virtual void setCanScroll(bool canScroll);

  /// Returns the current size of the web page.
  virtual int getWidth() const;
  virtual int getHeight() const;

  /// Waits for the page to load properly. This function should be called, before displaying the
  /// page.
  virtual void waitForFinishedLoading() const;

  /// Reloads the page.
  ///
  /// @param ignoreCache If set to true the site will not be using cached data.
  virtual void reload(bool ignoreCache) const;

  /// When the user clicked on a hyperlink on the web page, this functionality can be used to move
  /// forward or backwards in the history.
  virtual void goBack() const;
  virtual void goForward() const;

  /// These could be executed when the according hot keys are pressed. The system's clipboard will
  /// be used. So the user can actually copy-paste from one WebView to another or even from a
  /// WebView to third-party applications.
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
  /// This ensures statically that all given template types are either bool, double, std::string or
  /// std::string&&.
  template <typename... Args>
  static constexpr void assertJavaScriptTypes() {
    CS_WARNINGS_PUSH
    CS_DISABLE_GCC_WARNING("-Wunused-value")
    // Call assertJavaScriptType() for each Arg of Args.
    std::initializer_list<int>{(assertJavaScriptType<Args>(), 0)...};
    CS_WARNINGS_POP
  }

  /// This ensures statically that the given template type is either bool, double, std::string or
  /// std::string&&.
  template <typename T>
  static constexpr void assertJavaScriptType() {
    static_assert(
        std::is_same<T, double>() || std::is_same<T, bool>() || std::is_same<T, std::string>() ||
            std::is_same<T, std::string&&>() || std::is_same<T, std::optional<double>>() ||
            std::is_same<T, std::optional<bool>>() || std::is_same<T, std::optional<std::string>>(),
        "Only doubles, bools and std::strings are supported for JavaScript callback parameters "
        "(and std::optionals thereof)!");
  }

  /// The UnderlyingValue struct is used to access the actual value in a std::optional<JSType>.
  /// There are two variants of the struct as we may want to have the actual value (a bool, double
  /// or std::string) contained in the std::optional<JSType>, or a std::optional thereof (either a
  /// std::optional<bool>, std::optional<double>, or std::optional<std::string>).
  template <typename T>
  struct UnderlyingValue {
    static inline constexpr T get(std::optional<JSType>&& value) {
      return std::get<typename std::remove_reference<T>::type>(std::move(value.value()));
    }
  };

  template <typename T>
  struct UnderlyingValue<std::optional<T>> {
    static inline constexpr std::optional<T> get(std::optional<JSType>&& value) {
      if (value.has_value()) {
        return std::get<T>(std::move(value.value()));
      }
      return std::nullopt;
    }
  };

  /// This wraps the given callback in a lambda which will be stored in an internal map. This lambda
  /// receives its arguments as a std::vector<std::optional<JSType>>, each item in this vector will
  /// be casted to the required paramater types of the given callback.
  template <typename... Args, std::size_t... Is>
  void registerCallbackWrapper(std::string const& name, std::string const& comment,
      std::function<void(Args...)> const& callback, std::index_sequence<Is...> /*unused*/) {

    // The types vector is required to name the JavaScript function's arguments depending on its
    // type.
    std::vector<std::type_index> types = {std::type_index(typeid(Args))...};

    registerJSCallbackImpl(name, comment, std::move(types),
        [name, callback](std::vector<std::optional<JSType>>&& args) {
          // It is possible that the JavaScript method was called with less arguments than we expect
          // (if some of our arguments are optional). Therefore we pad the args vector with
          // std::nullopts.
          args.resize(sizeof...(Args));

          try {
            // Now call the actual callback. The UnderlyingValue struct is used to access the actual
            // value in the std::optional<JSType>. See its implementation above.
            callback(UnderlyingValue<Args>::get(std::move(args[Is]))...);
          } catch (std::exception const& e) {
            logger().error("Cannot execute javascript callback '{}': {}!", name, e.what());
          }
        });
  }

  void callJavascriptImpl(std::string const& function, std::vector<std::string> const& args) const;
  void registerJSCallbackImpl(std::string const& name, std::string const& comment,
      std::vector<std::type_index>&&                                   types,
      std::function<void(std::vector<std::optional<JSType>>&&)> const& callback);

  detail::WebViewClient* mClient;
  CefRefPtr<CefBrowser>  mBrowser;

  bool mInteractive = true;
  bool mCanScroll   = true;

  // Input state.
  int mMouseX         = 0;
  int mMouseY         = 0;
  int mMouseModifiers = 0;

  // Time point for the last left mouse click
  std::chrono::steady_clock::time_point mLastClick;

  // Count number of left mouse button clicks
  int mClickCount = 1;
};

} // namespace cs::gui

#endif // CS_GUI_WEBVIEW_HPP
