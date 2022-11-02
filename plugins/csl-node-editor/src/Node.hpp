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

  /// Returns true if any input connection has new data available. This will return true until
  /// readInput() is called once for the respective input connection.
  bool hasNewInput() const;

  /// Returns true if the given input connection has new data available. This will return true until
  /// readInput() is called once for the respective input connection.
  bool hasNewInput(std::string const& socket) const;

  /// Returns true if there is a (new) output connection which has never been written to.
  bool hasUndefinedOutput() const;

  /// Returns true if the given output connection has never been written to. This can happen if the
  /// output socket is freshly connected to another node.
  bool hasUndefinedOutput(std::string const& socket) const;

  template <typename T>
  void writeOutput(std::string const& socket, T const& value) {
    auto connections = mGraph->getOutputConnections(mID, socket);

    for (auto& c : connections) {
      if (!c->mData.has_value() || std::any_cast<T>(c->mData) != value) {
        c->mHasNewData = true;
        c->mData       = value;
      }
    }
  }

  template <typename T>
  T readInput(std::string const& socket, T defaultValue) {
    auto connection = mGraph->getInputConnection(mID, socket);

    if (connection && connection->mData.has_value()) {
      connection->mHasNewData = false;
      return std::any_cast<T>(connection->mData);
    }

    return std::move(defaultValue);
  }

  uint32_t                   mID;
  std::shared_ptr<NodeGraph> mGraph;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_HPP
