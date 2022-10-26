////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "NodeEditor.hpp"

#include "logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <CivetServer.h>
#include <functional>
#include <iostream>

namespace {

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

namespace csl::nodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

NodeEditor::NodeEditor(uint16_t port, NodeFactory factory)
    : mFactory(std::move(factory))
    , mHTMLSource(std::move(createHTMLSource())) {

  mHandlers.emplace("/", std::make_unique<GetHandler>([this](mg_connection* conn) {
    mg_send_http_ok(conn, "text/html", mHTMLSource.length());
    mg_write(conn, mHTMLSource.data(), mHTMLSource.length());
  }));

  mHandlers.emplace("**.css$", std::make_unique<GetHandler>([this](mg_connection* conn) {
    auto info = mg_get_request_info(conn);
    mg_send_mime_file(
        conn, ("../share/resources/gui/" + std::string(info->request_uri)).c_str(), "text/css");
  }));

  mHandlers.emplace("**.js$", std::make_unique<GetHandler>([this](mg_connection* conn) {
    auto info = mg_get_request_info(conn);
    mg_send_mime_file(conn, ("../share/resources/gui/" + std::string(info->request_uri)).c_str(),
        "text/javascript");
  }));

  mHandlers.emplace("**.ttf$", std::make_unique<GetHandler>([this](mg_connection* conn) {
    auto info = mg_get_request_info(conn);
    mg_send_mime_file(
        conn, ("../share/resources/gui/" + std::string(info->request_uri)).c_str(), "font/ttf");
  }));

  startServer(port);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

NodeEditor::~NodeEditor() {
  quitServer();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::startServer(uint16_t port) {

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

void NodeEditor::quitServer() {
  try {
    if (mServer) {
      mServer.reset();
    }
  } catch (std::exception const& e) { logger().warn("Failed to quit server: {}!", e.what()); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string NodeEditor::createHTMLSource() const {
  auto html = cs::utils::filesystem::loadToString("../share/resources/gui/csl-node-editor.html");

  cs::utils::replaceString(html, "//!SOCKET_SOURCE_CODE", mFactory.getSocketSource());
  cs::utils::replaceString(html, "//!NODE_SOURCE_CODE", mFactory.getNodeSource());
  cs::utils::replaceString(html, "//!REGISTER_COMPONENTS", mFactory.getRegisterSource());

  return html;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::nodeeditor
