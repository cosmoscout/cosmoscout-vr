////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WebSocket.hpp"

#include "logger.hpp"

#include <optional>

namespace csl::nodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::string> WebSocket::getNextEvent() {
  std::unique_lock<std::mutex> lock(mEventQueueMutex);
  if (!mEventQueue.empty()) {
    auto event = mEventQueue.front();
    mEventQueue.pop();
    return event;
  }

  return std::nullopt;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebSocket::sendData(std::string const& data) const {
  // mg_websocket_write(conn, MG_WEBSOCKET_OPCODE_TEXT, text, strlen(text));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebSocket::handleConnection(CivetServer* server, const struct mg_connection* conn) {
  csl::nodeeditor::logger().info("WS connected");
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebSocket::handleReadyState(CivetServer* server, struct mg_connection* conn) {
  csl::nodeeditor::logger().info("WS ready");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebSocket::handleData(
    CivetServer* server, struct mg_connection* conn, int bits, char* data, size_t data_len) {

  std::unique_lock<std::mutex> lock(mEventQueueMutex);
  mEventQueue.emplace(data, data_len);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebSocket::handleClose(CivetServer* server, const struct mg_connection* conn) {
  csl::nodeeditor::logger().info("WS closed");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::nodeeditor
