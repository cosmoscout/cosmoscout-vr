////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "NodeEditor.hpp"

#include "Node.hpp"
#include "logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <CivetServer.h>
#include <functional>
#include <iostream>
#include <optional>
#include <queue>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class EventType { eCustom, eAddNode, eRemoveNode, eAddConnection, eRemoveConnection };

// clang-format off
NLOHMANN_JSON_SERIALIZE_ENUM(EventType, {
    {EventType::eCustom,           "custom"},
    {EventType::eAddNode,          "addNode"},
    {EventType::eRemoveNode,       "removeNode"},
    {EventType::eAddConnection,    "addConnection"},
    {EventType::eRemoveConnection, "removeConnection"},
})
// clang-format on

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

class WebSocketHandler : public CivetWebSocketHandler {

 public:
  std::optional<std::string> getNextEvent() {
    std::unique_lock<std::mutex> lock(mEventQueueMutex);
    if (!mEventQueue.empty()) {
      auto event = mEventQueue.front();
      mEventQueue.pop();
      return event;
    }

    return std::nullopt;
  }

  void sendData(std::string const& data) const {
    // mg_websocket_write(conn, MG_WEBSOCKET_OPCODE_TEXT, text, strlen(text));
  }

 private:
  bool handleConnection(CivetServer* server, const struct mg_connection* conn) override {
    csl::nodeeditor::logger().info("WS connected");
    return true;
  }

  void handleReadyState(CivetServer* server, struct mg_connection* conn) override {
    csl::nodeeditor::logger().info("WS ready");
  }

  bool handleData(CivetServer* server, struct mg_connection* conn, int bits, char* data,
      size_t data_len) override {

    std::unique_lock<std::mutex> lock(mEventQueueMutex);
    mEventQueue.emplace(data, data_len);

    return true;
  }

  void handleClose(CivetServer* server, const struct mg_connection* conn) override {
    csl::nodeeditor::logger().info("WS closed");
  }

  std::queue<std::string> mEventQueue;
  std::mutex              mEventQueueMutex;
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
    , mGraph(std::make_shared<NodeGraph>())
    , mSocket(std::make_unique<WebSocketHandler>())
    , mHTMLSource(std::move(createHTMLSource())) {

  mHandlers.emplace_back("**.css$", std::make_unique<GetHandler>([this](mg_connection* conn) {
    auto info = mg_get_request_info(conn);
    mg_send_mime_file(
        conn, ("../share/resources/gui/" + std::string(info->request_uri)).c_str(), "text/css");
  }));

  mHandlers.emplace_back("**.js$", std::make_unique<GetHandler>([this](mg_connection* conn) {
    auto info = mg_get_request_info(conn);
    mg_send_mime_file(conn, ("../share/resources/gui/" + std::string(info->request_uri)).c_str(),
        "text/javascript");
  }));

  mHandlers.emplace_back("**.ttf$", std::make_unique<GetHandler>([this](mg_connection* conn) {
    auto info = mg_get_request_info(conn);
    mg_send_mime_file(
        conn, ("../share/resources/gui/" + std::string(info->request_uri)).c_str(), "font/ttf");
  }));

  mHandlers.emplace_back("/favicon.ico$", std::make_unique<GetHandler>([this](mg_connection* conn) {
    mg_send_mime_file(conn, "../share/resources/icons/icon.ico", "image/ico");
  }));

  mHandlers.emplace_back("/$", std::make_unique<GetHandler>([this](mg_connection* conn) {
    mg_send_http_ok(conn, "text/html", mHTMLSource.length());
    mg_write(conn, mHTMLSource.data(), mHTMLSource.length());
  }));

  startServer(port);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

NodeEditor::~NodeEditor() {
  quitServer();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::update() {
  auto socket = dynamic_cast<WebSocketHandler*>(mSocket.get());

  auto event = socket->getNextEvent();

  while (event) {

    try {

      logger().debug(event.value());

      nlohmann::json json = nlohmann::json::parse(event.value());

      EventType type = json.at("eventType");

      switch (type) {
      case EventType::eCustom:
        handleCustomEvent(json.at("data"));
        break;
      case EventType::eAddNode:
        handleAddNodeEvent(json.at("node"));
        break;
      case EventType::eRemoveNode:
        handleRemoveNodeEvent(json.at("node"));
        break;
      case EventType::eAddConnection:
        handleAddConnectionEvent(json.at("connection"));
        break;
      case EventType::eRemoveConnection:
        handleRemoveConnectionEvent(json.at("connection"));
        break;
      }

    } catch (std::exception const& e) {
      logger().error("Failed to process node editor event '{}': {}", event.value(), e.what());
    }

    event = socket->getNextEvent();
  }

  mGraph->process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::startServer(uint16_t port) {

  // First quit the server as it may be running already.
  quitServer();

  try {
    // We start the server with one thread only, as we do not want to process requests in parallel.
    std::vector<std::string> options{"listening_ports", std::to_string(port), "num_threads", "1"};
    mServer = std::make_unique<CivetServer>(options);
    mServer->addWebSocketHandler("/socket", *mSocket);

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

void NodeEditor::handleCustomEvent(nlohmann::json const& json) {
  logger().info("custom");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::handleAddNodeEvent(nlohmann::json const& json) {
  std::string type = json.at("name");
  uint32_t    id   = json.at("id");

  auto node = mFactory.createNode(type);
  node->setGraph(mGraph);
  node->setID(id);

  mGraph->addNode(id, std::move(node));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::handleRemoveNodeEvent(nlohmann::json const& json) {
  uint32_t id = json.at("id");

  mGraph->removeNode(id);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::handleAddConnectionEvent(nlohmann::json const& json) {
  uint32_t    fromNode   = json["input"]["connections"][0]["node"];
  std::string fromSocket = json["input"]["connections"][0]["output"];
  uint32_t    toNode     = json["output"]["connections"][0]["node"];
  std::string toSocket   = json["output"]["connections"][0]["input"];

  mGraph->addConnection(fromNode, fromSocket, toNode, toSocket);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::handleRemoveConnectionEvent(nlohmann::json const& json) {
  uint32_t    fromNode   = json["input"]["connections"][0]["node"];
  std::string fromSocket = json["input"]["connections"][0]["output"];
  uint32_t    toNode     = json["output"]["connections"][0]["node"];
  std::string toSocket   = json["output"]["connections"][0]["input"];

  mGraph->removeConnection(fromNode, fromSocket, toNode, toSocket);
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
