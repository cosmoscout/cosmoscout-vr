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

      switch (event.value().mType) {
      case WebSocket::Event::Type::eConnectionEstablished:
        mSocket->sendEvent({WebSocket::Event::Type::eLoadGraph, mGraph->toJSON()});
        logger().info("New node editor client connected.");
        break;
      case WebSocket::Event::Type::eConnectionDropped:
        logger().info("Node editor client disconnected.");
        break;
      case WebSocket::Event::Type::eGraphLoaded:
        mGraph->queueProcess();
        break;
      case WebSocket::Event::Type::eAddNode:
        handleAddNodeEvent(event.value().mData);
        break;
      case WebSocket::Event::Type::eRemoveNode:
        handleRemoveNodeEvent(event.value().mData);
        break;
      case WebSocket::Event::Type::eTranslateNode:
        handleTranslateNodeEvent(event.value().mData);
        break;
      case WebSocket::Event::Type::eCollapseNode:
        handleCollapseNodeEvent(event.value().mData);
        break;
      case WebSocket::Event::Type::eAddConnection:
        handleAddConnectionEvent(event.value().mData);
        break;
      case WebSocket::Event::Type::eRemoveConnection:
        handleRemoveConnectionEvent(event.value().mData);
        break;
      case WebSocket::Event::Type::eNodeMessage:
        handleNodeMessageEvent(event.value().mData);
        break;
      default:
        break;
      }

    } catch (std::exception const& e) {
      logger().error(
          "Failed to process node editor event '{}': {}", event.value().mData.dump(), e.what());
    }

    event = mSocket->getNextEvent();
  }

  try {
    mGraph->process();
  } catch (std::exception const& e) {
    logger().error("Failed to process node graph: {}", e.what());
  }
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
  std::string            type     = json["type"];
  uint32_t               id       = json["id"];
  std::array<int32_t, 2> position = json["position"];

  auto node = mFactory.createNode(type);
  node->setID(id);
  node->setPosition(position);
  node->setGraph(mGraph);
  node->setSocket(mSocket);

  mGraph->addNode(id, std::move(node));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::handleRemoveNodeEvent(nlohmann::json const& json) {
  mGraph->removeNode(json["id"]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::handleTranslateNodeEvent(nlohmann::json const& json) {
  uint32_t               id       = json["id"];
  std::array<int32_t, 2> position = json["position"];

  mGraph->setNodePosition(id, std::move(position));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::handleCollapseNodeEvent(nlohmann::json const& json) {
  uint32_t id        = json["id"];
  bool     collapsed = json["collapsed"];

  mGraph->setNodeCollapsed(id, collapsed);
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

void NodeEditor::handleNodeMessageEvent(nlohmann::json const& json) {
  uint32_t       toNode  = json["toNode"];
  nlohmann::json message = json["message"];
  mGraph->handleNodeMessage(toNode, message);
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
