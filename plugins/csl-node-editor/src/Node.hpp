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
#include <optional>
#include <string>
#include <vector>

namespace csl::nodeeditor {

class NodeGraph;
class Connection;

class CSL_NODE_EDITOR_EXPORT Node {
 public:
  Node()          = default;
  virtual ~Node() = default;

  /// This will be called whenever the values of one or multiple input sockets have changed.
  virtual void process(){};

  // virtual void onMessage(std::string const& data){};

  void setID(uint32_t id);
  void setGraph(std::shared_ptr<NodeGraph> graph);

 protected:
  // void sendMessage(std::string const& data) const;

  template <typename T>
  void writeOutput(std::string const& socket, T const& value) {
    auto connections = mGraph->getOutputConnections(mID, socket);

    for (auto& c : connections) {
      if (!c->mValue.has_value() || std::any_cast<T>(c->mValue) != value) {
        c->mHasNewData = true;
        c->mValue      = value;
      }
    }
  }

  template <typename T>
  std::optional<T> readInput(std::string const& socket) {
    auto connection = mGraph->getInputConnection(mID, socket);

    if (connection && connection->mValue.has_value()) {
      connection->mHasNewData = false;
      return std::any_cast<T>(connection->mValue);
    }

    return std::nullopt;
  }

 private:
  uint32_t                   mID;
  std::shared_ptr<NodeGraph> mGraph;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_HPP
