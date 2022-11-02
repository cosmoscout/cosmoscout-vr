////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_WEB_SOCKET_HPP
#define CSL_NODE_EDITOR_WEB_SOCKET_HPP

#include "csl_node_editor_export.hpp"

#include <CivetServer.h>
#include <mutex>
#include <queue>

namespace csl::nodeeditor {

class CSL_NODE_EDITOR_EXPORT WebSocket : public CivetWebSocketHandler {

 public:
  std::optional<std::string> getNextEvent();

  void sendMessage(std::string const& data) const;

 private:
  bool handleConnection(CivetServer* server, const struct mg_connection* conn) override;

  void handleReadyState(CivetServer* server, struct mg_connection* conn) override;

  bool handleData(CivetServer* server, struct mg_connection* conn, int bits, char* data,
      size_t data_len) override;

  void handleClose(CivetServer* server, const struct mg_connection* conn) override;

  std::queue<std::string> mEventQueue;
  std::mutex              mEventQueueMutex;

  mg_connection* mConnection = nullptr;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_WEB_SOCKET_HPP
