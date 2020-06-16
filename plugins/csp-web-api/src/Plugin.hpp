////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WEB_API_PLUGIN_HPP
#define CSP_WEB_API_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <optional>
#include <queue>
#include <unordered_map>

class CivetServer;
class CivetHandler;

namespace csp::webapi {

/// This plugin contains a web server which provides some HTTP endpoints which can be used to
/// remote-control CosmoScout VR.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    /// The port where the server should listen on. For example 9999.
    cs::utils::Property<uint16_t> mPort;

    /// You can provide a path to an html file which will be served when a GET request is sent to
    /// localhost:mPort, for example by a web browser. The path must be relative to the cosmoscout
    /// executable. Note that no other files are served by the server, so the given html file should
    /// not depend on other local resources.
    std::optional<std::string> mPage;
  };

  void init() override;
  void deInit() override;

  void update() override;

 private:
  void startServer(uint16_t port);
  void quitServer();

  Settings                                                       mPluginSettings;
  std::unique_ptr<CivetServer>                                   mServer;
  std::unordered_map<std::string, std::unique_ptr<CivetHandler>> mHandlers;

  // Members for the /capture endpoint
  std::mutex              mCaptureMutex;
  std::condition_variable mCaptureDone;
  bool                    mCaptureRequested = false;
  int32_t                 mCaptureWidth     = 0;
  int32_t                 mCaptureHeight    = 0;
  int32_t                 mCaptureDelay     = 0;
  std::string             mCaptureGui       = "auto";
  bool                    mCaptureDepth     = false;
  std::string             mCaptureFormat;
  int32_t                 mCaptureAtFrame = 0;
  std::vector<std::byte>  mCapture;

  // Members for the /log endpoint
  std::mutex              mLogMutex;
  std::deque<std::string> mLogMessages;

  // Members for the /save endpoint
  std::mutex              mSaveMutex;
  std::condition_variable mSaveDone;
  bool                    mSaveRequested = false;
  std::string             mSaveSettings;

  // Members for the /load endpoint
  std::mutex  mLoadMutex;
  std::string mLoadSettings;

  // Members for the /run-js endpoint
  std::mutex              mJavaScriptCallsMutex;
  std::queue<std::string> mJavaScriptCalls;

  int  mOnLoadConnection       = -1;
  int  mOnSaveConnection       = -1;
  int  mOnLogMessageConnection = -1;
  bool mReloadRequired         = true;
};

} // namespace csp::webapi

#endif // CSP_WEB_API_PLUGIN_HPP
