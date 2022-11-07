////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "NodeEditor.hpp"

#include "Node.hpp"
#include "NodeGraph.hpp"
#include "WebSocket.hpp"
#include "logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <CivetServer.h>
#include <functional>
#include <iostream>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class EventType { eAddNode, eRemoveNode, eAddConnection, eRemoveConnection, eNodeMessage };

// clang-format off
NLOHMANN_JSON_SERIALIZE_ENUM(EventType, {
    {EventType::eAddNode,          "addNode"},
    {EventType::eRemoveNode,       "removeNode"},
    {EventType::eAddConnection,    "addConnection"},
    {EventType::eRemoveConnection, "removeConnection"},
    {EventType::eNodeMessage,      "nodeMessage"},
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

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csl::nodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

NodeEditor::NodeEditor(uint16_t port, NodeFactory factory)
    : mFactory(std::move(factory))
    , mSocket(std::make_shared<WebSocket>())
    , mGraph(std::make_shared<NodeGraph>())
    , mHTMLSource(std::move(createHTMLSource())) {

  logger().debug(mHTMLSource);

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

  mHandlers.emplace_back("**.woff2$", std::make_unique<GetHandler>([this](mg_connection* conn) {
    auto info = mg_get_request_info(conn);
    mg_send_mime_file(
        conn, ("../share/resources/gui/" + std::string(info->request_uri)).c_str(), "font/woff2");
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
  auto event = mSocket->getNextEvent();

  while (event) {

    try {

      nlohmann::json json = nlohmann::json::parse(event.value());

      EventType type = json.at("eventType");

      switch (type) {
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
      case EventType::eNodeMessage:
        handleNodeMessageEvent(json.at("toNode"), json.at("data"));
        break;
      }

    } catch (std::exception const& e) {
      logger().error("Failed to process node editor event '{}': {}", event.value(), e.what());
    }

    event = mSocket->getNextEvent();
  }

  mGraph->process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::startServer(uint16_t port) {

  // First quit the server as it may be running already.
  quitServer();

  try {
    std::vector<std::string> options{"listening_ports", std::to_string(port)};
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

void NodeEditor::handleAddNodeEvent(nlohmann::json const& json) {
  std::string type = json["type"];
  uint32_t    id   = json["id"];

  auto node = mFactory.createNode(type);
  node->setGraph(mGraph);
  node->setSocket(mSocket);
  node->setID(id);

  mGraph->addNode(id, std::move(node));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::handleRemoveNodeEvent(nlohmann::json const& json) {
  mGraph->removeNode(json["id"]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::handleAddConnectionEvent(nlohmann::json const& json) {
  uint32_t    fromNode   = json["fromNode"];
  std::string fromSocket = json["fromSocket"];
  uint32_t    toNode     = json["toNode"];
  std::string toSocket   = json["toSocket"];

  mGraph->addConnection(fromNode, fromSocket, toNode, toSocket);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::handleRemoveConnectionEvent(nlohmann::json const& json) {
  uint32_t    fromNode   = json["fromNode"];
  std::string fromSocket = json["fromSocket"];
  uint32_t    toNode     = json["toNode"];
  std::string toSocket   = json["toSocket"];

  mGraph->removeConnection(fromNode, fromSocket, toNode, toSocket);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::handleNodeMessageEvent(uint32_t toNode, nlohmann::json const& json) {
  mGraph->handleNodeMessage(toNode, json);
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
