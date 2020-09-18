////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-scene/CelestialObserver.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "logger.hpp"

#include <CivetServer.h>
#include <GL/glew.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/DisplayManager/VistaWindow.h>
#include <VistaKernel/VistaFrameLoop.h>
#include <VistaKernel/VistaSystem.h>
#include <curlpp/cURLpp.hpp>
#include <sstream>
#include <tiffio.h>
#include <tiffio.hxx>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <utility>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::webapi::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Appends void* data to a std::vector<std::byte> (which is given through a void* as well). So this
// is pretty unsafe, but I think it's the only way to make stb_image write to a
// std::vector<std::byte>.
void stbWriteToVector(void* context, void* data, int len) {
  auto* vector   = static_cast<std::vector<std::byte>*>(context);
  auto* charData = static_cast<std::byte*>(data);
  // NOLINTNEXTLINE (cppcoreguidelines-pro-bounds-pointer-arithmetic)
  vector->insert(vector->end(), charData, charData + len);
}

// Encodes the pixel data in "in" to a in-memory tiff in "out".
template <typename T>
void tiffWriteToVector(std::vector<std::byte>& out, std::vector<T>& in, uint32_t width,
    uint32_t height, uint32_t samples, uint32_t bits) {

  std::ostringstream oStream;
  TIFF*              tiff = TIFFStreamOpen("MemTIFF", &oStream);

  TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, samples);
  TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bits);
  TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, 16);
  TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);

  if (samples == 3) {
    TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  } else {
    TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  }

  for (uint32_t i(0); i < height; ++i) {
    TIFFWriteScanline(tiff, &in.at((height - i - 1) * width * samples), i);
  }

  TIFFClose(tiff);

  // Convert the stringstream to a std::vector<std::byte>.
  std::string s = oStream.str();
  out.reserve(s.size());

  std::transform(s.begin(), s.end(), std::back_inserter(out),
      [](char& c) { return static_cast<std::byte>(c); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// A simple wrapper class which basically allows registering of lambdas as endpoint handlers for
// our CivetServer. This one handles GET requests.
class GetHandler : public CivetHandler {
 public:
  explicit GetHandler(std::function<void(mg_connection*)> handler)
      : mHandler(std::move(handler)) {
  }

  bool handleGet(CivetServer* /*server*/, mg_connection* conn) override {
    mHandler(conn);
    return true;
  }

 private:
  std::function<void(mg_connection*)> mHandler;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// A simple wrapper class which basically allows registering of lambdas as endpoint handlers for
// our CivetServer. This one handles POST requests.
class PostHandler : public CivetHandler {
 public:
  explicit PostHandler(std::function<void(mg_connection*)> handler)
      : mHandler(std::move(handler)) {
  }

  bool handlePost(CivetServer* /*server*/, mg_connection* conn) override {
    mHandler(conn);
    return true;
  }

 private:
  std::function<void(mg_connection*)> mHandler;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// A small helper method which returns the value of a parameter from a request URL. If the parameter
// is not present, a given default value is returned.
template <typename T>
T getParam(mg_connection* conn, std::string const& name, T const& defaultValue) {
  std::string s;
  if (CivetServer::getParam(conn, name.c_str(), s)) {
    return cs::utils::fromString<T>(s);
  }
  return defaultValue;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::webapi {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "port", o.mPort);
  cs::core::Settings::deserialize(j, "page", o.mPage);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "port", o.mPort);
  cs::core::Settings::serialize(j, "page", o.mPage);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  // We store all emitted log messages (up to a maximum of 1000) in a std::deque in order to be able
  // to answer to /log requests.
  mOnLogMessageConnection = cs::utils::onLogMessage().connect(
      [this](
          std::string const& logger, spdlog::level::level_enum level, std::string const& message) {
        const std::unordered_map<spdlog::level::level_enum, std::string> mapping = {
            {spdlog::level::trace, "T"}, {spdlog::level::debug, "D"}, {spdlog::level::info, "I"},
            {spdlog::level::warn, "W"}, {spdlog::level::err, "E"}, {spdlog::level::critical, "C"}};

        std::lock_guard<std::mutex> lock(mLogMutex);
        mLogMessages.push_front("[" + mapping.at(level) + "] " + logger + message);

        if (mLogMessages.size() > 1000) {
          mLogMessages.pop_back();
        }
      });

  // Return the landing page when the root document is requested. If not landing page is configured,
  // we just send back a simple message.
  mHandlers.emplace("/", std::make_unique<GetHandler>([this](mg_connection* conn) {
    if (mPluginSettings.mPage) {
      mg_send_mime_file(conn, mPluginSettings.mPage.value().c_str(), "text/html");
    } else {
      std::string response = "CosmoScout VR is running. You can modify this page with "
                             "the 'page' key in the configuration of 'csp-web-api'.";
      mg_send_http_ok(conn, "text/plain", response.length());
      mg_write(conn, response.data(), response.length());
    }
  }));

  // Return a json array of log messages for /log requests.
  mHandlers.emplace("/log", std::make_unique<GetHandler>([this](mg_connection* conn) {
    auto           length = getParam<uint32_t>(conn, "length", 100U);
    nlohmann::json json;

    {
      std::lock_guard<std::mutex> lock(mLogMutex);
      auto                        it = mLogMessages.begin();
      while (json.size() < length && it != mLogMessages.end()) {
        json.push_back(*it);
        ++it;
      }
    }

    std::string response = json.dump();
    mg_send_http_ok(conn, "application/json", response.length());
    mg_write(conn, response.data(), response.length());
  }));

  // Return a json object containing the current scene settings.
  mHandlers.emplace("/save", std::make_unique<GetHandler>([this](mg_connection* conn) {
    // This string will contain the json data at the end of this method.
    std::string response;
    {
      std::unique_lock<std::mutex> lock(mSaveMutex);

      // This tells the main thread that a save request is pending.
      mSaveRequested = true;

      // Now we use a condition variable to wait for the save data. It is actually saved in the
      // Plugin::update() method further below.
      mSaveDone.wait(lock);

      response = mSaveSettings;
    }

    mg_send_http_ok(conn, "application/json", response.length());
    mg_write(conn, response.data(), response.length());
  }));

  // Allows uploading of the current scene settings.
  mHandlers.emplace("/load", std::make_unique<PostHandler>([this](mg_connection* conn) {
    std::lock_guard<std::mutex> lock(mLoadMutex);
    mLoadSettings = CivetServer::getPostData(conn);

    std::string response = "Done.\r\n";
    mg_send_http_ok(conn, "text/plain", response.length());
    mg_write(conn, response.data(), response.length());
  }));

  // The /capture endpoint is a little bit more involved. As it takes several frames for the
  // capture to be completed (first we have to resize CosmoScout's window to the requested size,
  // then we have to wait some frames so that everything is loaded properly), we have to do some
  // more synchronization here.
  mHandlers.emplace("/capture", std::make_unique<GetHandler>([this](mg_connection* conn) {
    // First acquire the lock to make sure the mCapture* members are not currently read by the
    // main thread.
    std::unique_lock<std::mutex> lock(mCaptureMutex);

    // Read all paramters.
    mCaptureDelay  = std::clamp(getParam<int32_t>(conn, "delay", 50), 1, 200);
    mCaptureWidth  = std::clamp(getParam<int32_t>(conn, "width", 0), 0, 2000);
    mCaptureHeight = std::clamp(getParam<int32_t>(conn, "height", 0), 0, 2000);
    mCaptureGui    = getParam<std::string>(conn, "gui", "auto");
    mCaptureDepth  = getParam<std::string>(conn, "depth", "false") == "true";
    mCaptureFormat = getParam<std::string>(conn, "format", mCaptureDepth ? "tiff" : "png");

    // Validate format parameter.
    if (mCaptureFormat != "png" && mCaptureFormat != "jpeg" && mCaptureFormat != "tiff") {
      mg_send_http_error(
          conn, 422, "Only 'png', 'jpeg', or 'tiff' are allowed for the format parameter!");
      return;
    }

    // Validate gui parameter.
    if (mCaptureGui != "auto" && mCaptureGui != "true" && mCaptureGui != "false") {
      mg_send_http_error(
          conn, 422, "Only 'auto', 'true', or 'false' are allowed for the gui parameter!");
      return;
    }

    // This tells the main thread that a capture request is pending.
    mCaptureRequested = true;

    // Now we use a condition variable to wait for the capture. It is actually captured in the
    // Plugin::update() method further below.
    mCaptureDone.wait(lock);

    // The capture has been captured, return the result!
    mg_send_http_ok(conn, ("image/" + mCaptureFormat).c_str(), mCapture.size());
    mg_write(conn, mCapture.data(), mCapture.size());
  }));

  // All POST requests received on /run-js are stored in a queue. They are executed in the main
  // thread in the Plugin::update() method further below.
  mHandlers.emplace("/run-js", std::make_unique<PostHandler>([this](mg_connection* conn) {
    std::lock_guard<std::mutex> lock(mJavaScriptCallsMutex);
    mJavaScriptCalls.push(CivetServer::getPostData(conn));

    std::string response = "Done.\r\n";
    mg_send_http_ok(conn, "text/plain", response.length());
    mg_write(conn, response.data(), response.length());
  }));

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { mReloadRequired = true; });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-web-api"] = mPluginSettings; });

  // Restart the server if the port changes.
  mPluginSettings.mPort.connect([this](uint16_t port) { startServer(port); });

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);
  cs::utils::onLogMessage().disconnect(mOnLogMessageConnection);

  quitServer();

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {

  // Execute all /run-js requests received since the last call to update().
  {
    std::lock_guard<std::mutex> lock(mJavaScriptCallsMutex);
    while (!mJavaScriptCalls.empty()) {
      auto request = mJavaScriptCalls.front();
      mJavaScriptCalls.pop();
      logger().debug("Executing '/run-js' request: '{}'", request);
      mGuiManager->getGui()->executeJavascript(request);
    }
  }

  // Execute any pending /save request.
  {
    std::lock_guard<std::mutex> lock(mSaveMutex);
    if (mSaveRequested) {
      logger().debug("Executing '/save' request.");
      try {
        mSaveSettings = mAllSettings->saveToJson();
      } catch (std::exception const& e) {
        logger().error("Failed to write settings: {}", e.what());
        mSaveSettings = "";
      }
      mSaveDone.notify_one();
      mSaveRequested = false;
    }
  }

  // Execute any pending /load request.
  {
    std::lock_guard<std::mutex> lock(mLoadMutex);
    if (!mLoadSettings.empty()) {
      logger().debug("Executing '/load' request.");
      try {
        mAllSettings->loadFromJson(mLoadSettings);
      } catch (std::exception const& e) {
        logger().error("Failed to read settings: {}", e.what());
        mSaveSettings = "";
      }
      mLoadSettings.clear();
    }
  }

  // If a screen shot has been requested, we first resize the image to the given size. Then we wait
  // mCaptureDelay frames until we actually read the pixels.
  {
    std::lock_guard<std::mutex> lock(mCaptureMutex);
    if (mCaptureRequested) {
      if (mCaptureWidth > 0 && mCaptureHeight > 0) {
        auto* window = GetVistaSystem()->GetDisplayManager()->GetWindows().begin()->second;
        window->GetWindowProperties()->SetSize(mCaptureWidth, mCaptureHeight);
      }
      mCaptureAtFrame = GetVistaSystem()->GetFrameLoop()->GetFrameCount() + mCaptureDelay;
      if (mCaptureGui != "auto") {
        mAllSettings->pEnableUserInterface = mCaptureGui == "true";
      }
      mCaptureRequested = false;
    }

    // Now we waited several frames. We read the pixels, encode the data as png and notify the
    // server's worker thread that the screen shot is done.
    if (mCaptureAtFrame > 0 &&
        mCaptureAtFrame == GetVistaSystem()->GetFrameLoop()->GetFrameCount()) {

      auto* window = GetVistaSystem()->GetDisplayManager()->GetWindows().begin()->second;
      window->GetWindowProperties()->GetSize(mCaptureWidth, mCaptureHeight);

      logger().debug("Capturing capture for /capture request: resolution = {}x{}, show gui = {}, "
                     "depth = {}, format = {}",
          mCaptureWidth, mCaptureHeight, mCaptureGui, mCaptureDepth, mCaptureFormat);

      // We encode image data in the main thread as this is not thread-safe.
      stbi_flip_vertically_on_write(1);

      // Clear last data first.
      mCapture.clear();

      if (mCaptureDepth) {

        // Capture the depth component.
        std::vector<float> capture(mCaptureWidth * mCaptureHeight);
        glReadPixels(
            0, 0, mCaptureWidth, mCaptureHeight, GL_DEPTH_COMPONENT, GL_FLOAT, &capture[0]);

        if (mCaptureFormat == "tiff") {
          // We retrieve the current scene scale and far-clip distance in order to scale the depth
          // values to meters.
          double      nearClip{};
          double      farClip{};
          auto const& p = *GetVistaSystem()->GetDisplayManager()->GetProjectionsConstRef().begin();
          p.second->GetProjectionProperties()->GetClippingRange(nearClip, farClip);

          auto scale = static_cast<float>(farClip * mSolarSystem->getObserver().getAnchorScale());
          for (auto& f : capture) {
            f *= scale;
          }

          // Now write the tiff image.
          tiffWriteToVector(mCapture, capture, mCaptureWidth, mCaptureHeight, 1, 32);

        } else {
          // Capture format is png or jpeg, let's convert the depth to 8-bit.
          std::vector<std::byte> captureByte(mCaptureWidth * mCaptureHeight);
          for (size_t i(0); i < capture.size(); ++i) {
            // The funny cast is required for MSVC 14.1 which does not like casting floating point
            // numbers to std::byte.
            captureByte[i] = static_cast<std::byte>(static_cast<uint8_t>(capture[i] * 255.0));
          }

          if (mCaptureFormat == "png") {
            stbi_write_png_to_func(&stbWriteToVector, &mCapture, mCaptureWidth, mCaptureHeight, 1,
                captureByte.data(), mCaptureWidth);
          } else {
            stbi_write_jpg_to_func(&stbWriteToVector, &mCapture, mCaptureWidth, mCaptureHeight, 1,
                captureByte.data(), 80);
          }
        }

      } else {

        // Capturing color images is pretty straight-forward.
        std::vector<std::byte> capture(mCaptureWidth * mCaptureHeight * 3);
        glReadPixels(0, 0, mCaptureWidth, mCaptureHeight, GL_RGB, GL_UNSIGNED_BYTE, &capture[0]);

        if (mCaptureFormat == "tiff") {
          tiffWriteToVector(mCapture, capture, mCaptureWidth, mCaptureHeight, 3, 8);
        } else if (mCaptureFormat == "png") {
          stbi_write_png_to_func(&stbWriteToVector, &mCapture, mCaptureWidth, mCaptureHeight, 3,
              capture.data(), mCaptureWidth * 3);
        } else {
          stbi_write_jpg_to_func(
              &stbWriteToVector, &mCapture, mCaptureWidth, mCaptureHeight, 3, capture.data(), 80);
        }
      }

      // Restore stb image state to default.
      stbi_flip_vertically_on_write(0);

      mCaptureAtFrame = 0;

      // The capture has been done, notify the worker thread,
      mCaptureDone.notify_one();
    }
  }

  // In this plugin, we cannot call this directly when the onLoad signal of the settings is fired,
  // since reloading can cause our server to be restarted. And as reloading can be triggered from a
  // /load request, this could lead to a deadlock.
  if (mReloadRequired) {
    from_json(mAllSettings->mPlugins.at("csp-web-api"), mPluginSettings);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::startServer(uint16_t port) {

  // First quit the server as it may be running already.
  quitServer();

  try {
    // We start the server with one thread only, as we do not want to process requests in parallel.
    std::vector<std::string> options{"listening_ports", std::to_string(port), "num_threads", "1"};
    mServer = std::make_unique<CivetServer>(options);

    for (auto const& handler : mHandlers) {
      mServer->addHandler(handler.first, *handler.second);
    }

  } catch (std::exception const& e) { logger().warn("Failed to start server: {}!", e.what()); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::quitServer() {
  try {
    if (mServer) {
      mServer.reset();
    }
  } catch (std::exception const& e) { logger().warn("Failed to quit server: {}!", e.what()); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::webapi
