////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_HPP
#define CSL_NODE_EDITOR_NODE_HPP

#include "csl_node_editor_export.hpp"

#include "NodeGraph.hpp"

#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace csl::nodeeditor {

class NodeGraph;
class WebSocket;
class Connection;

class CSL_NODE_EDITOR_EXPORT Node {
 public:
  Node()          = default;
  virtual ~Node() = default;

  /// This will be called whenever the values of one or multiple input sockets have changed.
  virtual void process() = 0;

  virtual void onMessageFromJS(nlohmann::json const& data){};

  void setID(uint32_t id);
  void setSocket(std::shared_ptr<WebSocket> socket);
  void setGraph(std::shared_ptr<NodeGraph> graph);

 protected:
  void sendMessageToJS(nlohmann::json const& data) const;

  template <typename T>
  void writeOutput(std::string const& socket, T const& value) {
    auto connections = mGraph->getOutputConnections(mID, socket);

    for (auto& c : connections) {
      if (!c->mData.has_value() || std::any_cast<T>(c->mData) != value) {
        mGraph->queueProcessing(c->mToNode);
        c->mData = value;
      }
    }
  }

  template <typename T>
  T readInput(std::string const& socket, T defaultValue) {
    auto connection = mGraph->getInputConnection(mID, socket);

    if (connection && connection->mData.has_value()) {
      return std::any_cast<T>(connection->mData);
    }

    return std::move(defaultValue);
  }

  uint32_t                   mID;
  std::shared_ptr<WebSocket> mSocket;
  std::shared_ptr<NodeGraph> mGraph;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_HPP
