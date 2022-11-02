////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_GRAPH_HPP
#define CSL_NODE_EDITOR_NODE_GRAPH_HPP

#include "Connection.hpp"

#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace csl::nodeeditor {

class Node;

/// This class keeps track of the nodes and their connections. It is used by the NodeEditor and the
/// Node base class. When implementing custom nodes, you usually will not have to work with this
/// class. Use the methods of the Node class instead.
class CSL_NODE_EDITOR_EXPORT NodeGraph {
 public:
  // These need to be declared explicitely as the default versions would be defined inline which
  // makes it impossible to use a forward declartion of Node.
  NodeGraph();
  ~NodeGraph();

  // Node API --------------------------------------------------------------------------------------

  // The methods below are primarily meant to be used by the Node base class. Usually, you should
  // not have to call them directly.

  /// Gets a connection which is connected to a given input socket. There can be at most one
  /// connection *to* a socket.
  /// @param toNode   The ID of the node.
  /// @param toSocket The name of the socket.
  /// @return         A connection (if any).
  Connection const* getInputConnection(uint32_t toNode, std::string const& toSocket) const;

  /// Gets all input connections connected to a given node.
  /// @param toNode   The ID of the node.
  /// @return         A list of connections (this can be empty).
  std::vector<Connection const*> getInputConnections(uint32_t toNode) const;

  /// Gets a list of connections which are connected to a given output socket. There can be multiple
  /// connections *from* a socket.
  /// @param fromNode   The ID of the node.
  /// @param fromSocket The name of the socket.
  /// @return           A list of connections (this can be empty).
  std::vector<Connection const*> getOutputConnections(
      uint32_t fromNode, std::string const& fromSocket) const;

  /// Gets all output connections connected to a given node.
  /// @param fromNode The ID of the node.
  /// @return         A list of connections (this can be empty).
  std::vector<Connection const*> getOutputConnections(uint32_t fromNode) const;

  // Node editor API -------------------------------------------------------------------------------

  // The methods below are primarily meant to be used by the NodeEditor class. Usually, you should
  // not have to call them.

  void process();

  void addNode(uint32_t id, std::unique_ptr<Node> node);

  void removeNode(uint32_t id);

  void addConnection(
      uint32_t fromNode, std::string fromSocket, uint32_t toNode, std::string toSocket);

  void removeConnection(uint32_t fromNode, std::string const& fromSocket, uint32_t toNode,
      std::string const& toSocket);

 private:
  // Returns all nodes which are currently connected to an output socket of the given node.
  std::vector<uint32_t> getOutputNodes(uint32_t node) const;

  // Returns all nodes which are currently connected to an input socket of the given node.
  std::vector<uint32_t> getInputNodes(uint32_t node) const;

  std::unordered_map<uint32_t, std::unique_ptr<Node>> mNodes;

  std::list<Connection> mConnections;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_GRAPH_HPP
