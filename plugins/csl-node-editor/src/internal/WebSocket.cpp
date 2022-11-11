////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WebSocket.hpp"

#include "../logger.hpp"

#include <optional>

namespace csl::nodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off
NLOHMANN_JSON_SERIALIZE_ENUM(WebSocket::Event::Type, {
    {WebSocket::Event::Type::eConnectionEstablished, "connectionEstablished"},
    {WebSocket::Event::Type::eConnectionDropped,     "connectionDropped"},
    {WebSocket::Event::Type::eLoadGraph,             "loadGraph"},
    {WebSocket::Event::Type::eGraphLoaded,           "graphLoaded"},
    {WebSocket::Event::Type::eAddNode,               "addNode"},
    {WebSocket::Event::Type::eRemoveNode,            "removeNode"},
    {WebSocket::Event::Type::eTranslateNode,         "translateNode"},
    {WebSocket::Event::Type::eCollapseNode,          "collapseNode"},
    {WebSocket::Event::Type::eAddConnection,         "addConnection"},
    {WebSocket::Event::Type::eRemoveConnection,      "removeConnection"},
    {WebSocket::Event::Type::eNodeMessage,           "nodeMessage"},
})
// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<WebSocket::Event> WebSocket::getNextEvent() {
  std::unique_lock<std::mutex> lock(mEventQueueMutex);
  if (!mEventQueue.empty()) {
    auto event = mEventQueue.front();
    mEventQueue.pop();
    return event;
  }

  return std::nullopt;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebSocket::sendEvent(WebSocket::Event const& event) const {
  if (isConnected()) {

    nlohmann::json json   = {{"type", event.mType}, {"data", event.mData}};
    auto           string = json.dump();

    mg_websocket_write(mConnection, MG_WEBSOCKET_OPCODE_TEXT, string.c_str(), string.size());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebSocket::isConnected() const {
  return mConnection != nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebSocket::handleConnection(CivetServer* server, const struct mg_connection* conn) {
  return !isConnected();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebSocket::handleReadyState(CivetServer* server, struct mg_connection* conn) {
  mConnection = conn;

  std::unique_lock<std::mutex> lock(mEventQueueMutex);
  mEventQueue.push({Event::Type::eConnectionEstablished});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebSocket::handleData(
    CivetServer* server, struct mg_connection* conn, int bits, char* data, size_t data_len) {

  if (data_len <= 4) {
    return true;
  }

  try {
    auto  json = nlohmann::json::parse(std::string(data, data_len));
    Event event{json["type"], json["data"]};

    std::unique_lock<std::mutex> lock(mEventQueueMutex);
    mEventQueue.push(event);
  } catch (std::exception const& e) {
    logger().warn("Failed to parse event: {}", std::string(data, data_len));
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebSocket::handleClose(CivetServer* server, const struct mg_connection* conn) {
  mConnection = nullptr;

  std::unique_lock<std::mutex> lock(mEventQueueMutex);
  mEventQueue.push({Event::Type::eConnectionDropped});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::nodeeditor
