////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "NodeEditor.hpp"

#include "Node.hpp"
#include "internal/CommunicationChannel.hpp"
#include "internal/NodeGraph.hpp"
#include "logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <CivetServer.h>
#include <functional>
#include <iostream>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

// A simple wrapper class which basically allows registering lambdas as GET endpoint handlers for
// our CivetServer.
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
    , mSocket(std::make_shared<CommunicationChannel>())
    , mGraph(std::make_shared<NodeGraph>())
    , mHTMLSource(std::move(createHTMLSource())) {

  std::vector<std::array<std::string, 2>> resourceHandlers{
      {"**.css$", "text/css"},
      {"**.js$", "text/javascript"},
      {"**.ttf$", "font/ttf"},
      {"**.woff2$", "font/woff2"},
  };

  for (auto const& h : resourceHandlers) {
    mHandlers.emplace_back(h[0], std::make_unique<GetHandler>([=](mg_connection* conn) {
      auto info = mg_get_request_info(conn);
      mg_send_mime_file(
          conn, ("../share/resources/gui/" + std::string(info->request_uri)).c_str(), h[1].c_str());
    }));
  }

  mHandlers.emplace_back("/favicon.ico$", std::make_unique<GetHandler>([this](mg_connection* conn) {
    mg_send_mime_file(conn, "../share/resources/icons/icon.ico", "image/ico");
  }));

  mHandlers.emplace_back("/$", std::make_unique<GetHandler>([this](mg_connection* conn) {
    mg_send_http_ok(conn, "text/html", mHTMLSource.length());
    mg_write(conn, mHTMLSource.data(), mHTMLSource.length());
  }));

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

NodeEditor::~NodeEditor() {
  try {
    if (mServer) {
      mServer.reset();
    }
  } catch (std::exception const& e) { logger().warn("Failed to quit server: {}!", e.what()); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::update() {

  // First, we process all events which we received via the communication channel from the connected
  // client (if there is any).
  auto event = mSocket->getNextEvent();

  while (event) {

    try {
      auto const& e = event.value();

      switch (e.mType) {

        // We will receive this event whenever a new client has been connected successfully. We send
        // the current graph to the client.
      case CommunicationChannel::Event::Type::eConnectionEstablished:
        mSocket->sendEvent({CommunicationChannel::Event::Type::eLoadGraph, mGraph->toJSON()});
        logger().info("New node editor client connected.");
        break;

        // We print a message when a connection is dropped.
      case CommunicationChannel::Event::Type::eConnectionDropped:
        logger().info("Node editor client disconnected.");
        break;

        // As soon as the client has loaded the graph which we have sent as answer to
        // eConnectionEstablished, we will receive this event. We will process the entire graph once
        // so that each node gets the chance to update its JavaScript counterpart.
      case CommunicationChannel::Event::Type::eGraphLoaded:
        mGraph->queueProcess();
        break;

        // We receive this event whenever the user created a new node at the frontend. We create a
        // corresponding C++ node.
      case CommunicationChannel::Event::Type::eAddNode: {
        std::string            type     = e.mData["type"];
        uint32_t               id       = e.mData["id"];
        std::array<int32_t, 2> position = e.mData["position"];

        auto node = mFactory.createNode(type);
        node->setID(id);
        node->setPosition(position);
        node->setGraph(mGraph);
        node->setSocket(mSocket);

        mGraph->addNode(id, std::move(node));
      } break;

        // If the user deleted a node, we delete its C++ counterpart as well.
      case CommunicationChannel::Event::Type::eRemoveNode:
        mGraph->removeNode(e.mData["id"]);
        break;

        // If the user moved a node around, we store the new position. This will be important when
        // restoring a graph layout later.
      case CommunicationChannel::Event::Type::eTranslateNode:
        mGraph->setNodePosition(e.mData["id"], e.mData["position"]);
        break;

        // If the user collapsed or un-collapsed a node, we store this information. This will be
        // important when restoring a graph layout later.
      case CommunicationChannel::Event::Type::eCollapseNode:
        mGraph->setNodeCollapsed(e.mData["id"], e.mData["collapsed"]);
        break;

        // We receive this event whenever the user created a new node connection at the frontend. We
        // create a corresponding C++ node connection.
      case CommunicationChannel::Event::Type::eAddConnection:
        mGraph->addConnection(
            e.mData["fromNode"], e.mData["fromSocket"], e.mData["toNode"], e.mData["toSocket"]);
        break;

        // If the user deleted a node connection, we delete its C++ counterpart as well.
      case CommunicationChannel::Event::Type::eRemoveConnection:
        mGraph->removeConnection(
            e.mData["fromNode"], e.mData["fromSocket"], e.mData["toNode"], e.mData["toSocket"]);
        break;

        // We receive this event whenever a JavaScript node sends a message to its C++ counterpart.
        // We let the node graph send this message to the receiver node.
      case CommunicationChannel::Event::Type::eNodeMessage:
        mGraph->handleNodeMessage(e.mData["toNode"], e.mData["message"]);
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

  // Finally, we will process the graph. This will call process() on all nodes which need to be
  // reprocessed in response to the events above.
  try {
    mGraph->process();
  } catch (std::exception const& e) {
    logger().error("Failed to process node graph: {}", e.what());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json NodeEditor::toJSON() const {
  return mGraph->toJSON();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeEditor::fromJSON(nlohmann::json const& json) {

  // First delete all nodes and connections.
  mGraph->clear();

  // Then create new nodes and connections according to the given structure.
  for (auto& [i, jsonNode] : json["nodes"].items()) {

    std::string            type      = jsonNode["name"];
    uint32_t               id        = jsonNode["id"];
    std::array<int32_t, 2> position  = jsonNode["position"];
    bool                   collapsed = jsonNode["collapsed"];

    auto node = mFactory.createNode(type);
    node->setID(id);
    node->setPosition(position);
    node->setIsCollapsed(collapsed);
    node->setGraph(mGraph);
    node->setSocket(mSocket);
    node->setData(jsonNode["data"]);

    mGraph->addNode(id, std::move(node));

    for (auto& [fromSocket, jsonOutput] : jsonNode["outputs"].items()) {
      for (auto& connection : jsonOutput["connections"]) {
        uint32_t    toNode   = connection["node"];
        std::string toSocket = connection["input"];

        mGraph->addConnection(id, fromSocket, toNode, toSocket);
      }
    }
  }

  // Finally, if there is already a connected client, send the new graph.
  mSocket->sendEvent({CommunicationChannel::Event::Type::eLoadGraph, json});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string NodeEditor::createHTMLSource() const {
  auto html = cs::utils::filesystem::loadToString("../share/resources/gui/csl-node-editor.html");

  // Replace the placeholders with the respective source code snippets.
  cs::utils::replaceString(html, "//!SOCKET_SOURCE_CODE", mFactory.getSocketSource());
  cs::utils::replaceString(html, "//!NODE_SOURCE_CODE", mFactory.getNodeSource());
  cs::utils::replaceString(html, "//!REGISTER_COMPONENTS", mFactory.getRegisterSource());

  return html;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::nodeeditor
