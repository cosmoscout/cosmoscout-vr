////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CommunicationChannel.hpp"

#include "../logger.hpp"

#include <optional>

namespace csl::nodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off
NLOHMANN_JSON_SERIALIZE_ENUM(CommunicationChannel::Event::Type, {
    {CommunicationChannel::Event::Type::eConnectionEstablished, "connectionEstablished"},
    {CommunicationChannel::Event::Type::eConnectionDropped,     "connectionDropped"},
    {CommunicationChannel::Event::Type::eLoadGraph,             "loadGraph"},
    {CommunicationChannel::Event::Type::eGraphLoaded,           "graphLoaded"},
    {CommunicationChannel::Event::Type::eAddNode,               "addNode"},
    {CommunicationChannel::Event::Type::eRemoveNode,            "removeNode"},
    {CommunicationChannel::Event::Type::eTranslateNode,         "translateNode"},
    {CommunicationChannel::Event::Type::eCollapseNode,          "collapseNode"},
    {CommunicationChannel::Event::Type::eAddConnection,         "addConnection"},
    {CommunicationChannel::Event::Type::eRemoveConnection,      "removeConnection"},
    {CommunicationChannel::Event::Type::eNodeMessage,           "nodeMessage"},
})
// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<CommunicationChannel::Event> CommunicationChannel::getNextEvent() {
  std::unique_lock<std::mutex> lock(mEventQueueMutex);
  if (!mEventQueue.empty()) {
    auto event = mEventQueue.front();
    mEventQueue.pop();
    return event;
  }

  return std::nullopt;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CommunicationChannel::sendEvent(CommunicationChannel::Event const& event) const {
  if (isConnected()) {

    nlohmann::json json   = {{"type", event.mType}, {"data", event.mData}};
    auto           string = json.dump();

    mg_websocket_write(mConnection, MG_WEBSOCKET_OPCODE_TEXT, string.c_str(), string.size());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CommunicationChannel::isConnected() const {
  return mConnection != nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CommunicationChannel::handleConnection(CivetServer* server, const struct mg_connection* conn) {
  return !isConnected();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CommunicationChannel::handleReadyState(CivetServer* server, struct mg_connection* conn) {
  mConnection = conn;

  std::unique_lock<std::mutex> lock(mEventQueueMutex);
  mEventQueue.push({Event::Type::eConnectionEstablished});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CommunicationChannel::handleData(
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
    logger().warn("Failed to parse event '{}': {}", std::string(data, data_len), e.what());
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CommunicationChannel::handleClose(CivetServer* server, const struct mg_connection* conn) {
  mConnection = nullptr;

  std::unique_lock<std::mutex> lock(mEventQueueMutex);
  mEventQueue.push({Event::Type::eConnectionDropped});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::nodeeditor
