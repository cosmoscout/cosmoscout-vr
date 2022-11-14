////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_GRAPH_HPP
#define CSL_NODE_EDITOR_NODE_GRAPH_HPP

#include "NodeConnection.hpp"

#include <list>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace csl::nodeeditor {

class Node;

/// This class keeps track of the nodes and their connections. It is used by the NodeEditor and the
/// Node base class. When implementing custom nodes, you usually will not have to work with this
/// class directly. Use the methods of the Node class instead.
class CSL_NODE_EDITOR_EXPORT NodeGraph {
 public:
  // These need to be declared explicitely as the default versions would be defined inline which
  // makes it impossible to use a forward declartion of Node.
  NodeGraph();
  ~NodeGraph();

  /// This will force each node to be processed during the next call to process().
  void queueProcess();

  /// This will force the given node to be processed during the next call to process(). There is
  /// usually no need to call this directly, as it is done by the Node base class whenever an output
  /// is written.
  /// @param node The ID of the node which should be processed.
  void queueProcess(uint32_t node);

  /// See NodeEditor::toJSON()
  nlohmann::json toJSON() const;

  // Node API --------------------------------------------------------------------------------------

  // The methods below are primarily meant to be used by the Node base class. Usually, you should
  // not have to call them directly.

  /// Gets a connection which is connected to a given input socket. There can be at most one
  /// connection *to* a socket.
  /// @param toNode   The ID of the node.
  /// @param toSocket The name of the socket.
  /// @return         A connection (if any).
  NodeConnection const* getInputConnection(uint32_t toNode, std::string const& toSocket) const;

  /// Gets all input connections connected to a given node.
  /// @param toNode   The ID of the node.
  /// @return         A list of connections (this can be empty).
  std::vector<NodeConnection const*> getInputConnections(uint32_t toNode) const;

  /// Gets a list of connections which are connected to a given output socket. There can be multiple
  /// connections *from* a socket.
  /// @param fromNode   The ID of the node.
  /// @param fromSocket The name of the socket.
  /// @return           A list of connections (this can be empty).
  std::vector<NodeConnection const*> getOutputConnections(
      uint32_t fromNode, std::string const& fromSocket) const;

  /// Gets all output connections connected to a given node.
  /// @param fromNode The ID of the node.
  /// @return         A list of connections (this can be empty).
  std::vector<NodeConnection const*> getOutputConnections(uint32_t fromNode) const;

  // Node editor API -------------------------------------------------------------------------------

  // The methods below are primarily meant to be used by the NodeEditor class. Usually, you should
  // not have to call them.

  /// Calls process() on all nodes which need a reprocessing. This could be due to changed input
  /// values, dropped input connections, new output connections, or due to the entire graph needing
  /// a reprocessing because a new web client connected.
  void process();

  /// Adds a new node to the graph. This is called by the NodeEditor class whenever the user adds a
  /// new node on the web frontend.
  /// @param id   The unique ID of the node.
  /// @param node The actual node instance.
  void addNode(uint32_t id, std::unique_ptr<Node> node);

  /// Removes a node from the graph. This is called by the NodeEditor class whenever the user
  /// removes a node on the web frontend.
  /// @param id   The ID of the node.
  void removeNode(uint32_t id);

  /// Updates the position of a node. This is called by the NodeEditor class whenever the user
  /// moves a node around on the web frontend.
  /// @param id        The ID of the moved node.
  /// @param position  The new position of the node.
  void setNodePosition(uint32_t id, std::array<int32_t, 2> position) const;

  /// Updates the collapsed state of a node. This is called by the NodeEditor class whenever the
  /// user collapses or un-collapses a node on the web frontend.
  /// @param id   The ID of the moved node.
  /// @param bool True if the node is currently collapsed.
  void setNodeCollapsed(uint32_t id, bool collapsed) const;

  /// Adds a new connection to the node graph. This is called by the NodeEditor class whenever the
  /// user connects an output socket to an input socket on the web frontend.
  /// @param fromNode   The ID of the input node.
  /// @param fromSocket The name of the input socket.
  /// @param toNode     The ID of the output node.
  /// @param toSocket   The name of the output socket.
  void addConnection(
      uint32_t fromNode, std::string fromSocket, uint32_t toNode, std::string toSocket);

  /// Removes a connection from the node graph. This is called by the NodeEditor class whenever the
  /// user disconnects two nodes from another on the web frontend.
  /// @param fromNode   The ID of the input node.
  /// @param fromSocket The name of the input socket.
  /// @param toNode     The ID of the output node.
  /// @param toSocket   The name of the output socket.
  void removeConnection(uint32_t fromNode, std::string const& fromSocket, uint32_t toNode,
      std::string const& toSocket);

  /// This is called by the node editor whenever a JavaScript node has sent some data to its C++
  /// counterpart. The node graph forwards the message to the respective node.
  /// @param toNode  The ID of the target node.
  /// @param message The custom message.
  void handleNodeMessage(uint32_t toNode, nlohmann::json const& message) const;

  /// This removes all nodes and connections from this node graph.
  void clear();

 private:
  // Returns all nodes which are currently connected to an output socket of the given node.
  std::vector<uint32_t> getOutputNodes(uint32_t node) const;

  // Returns all nodes which are currently connected to an input socket of the given node.
  std::vector<uint32_t> getInputNodes(uint32_t node) const;

  std::unordered_map<uint32_t, std::unique_ptr<Node>> mNodes;
  std::unordered_set<uint32_t>                        mDirtyNodes;
  std::list<NodeConnection>                           mConnections;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_GRAPH_HPP
