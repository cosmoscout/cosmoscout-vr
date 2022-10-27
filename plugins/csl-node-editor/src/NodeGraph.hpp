////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_GRAPH_HPP
#define CSL_NODE_EDITOR_NODE_GRAPH_HPP

#include "Connection.hpp"
#include "Node.hpp"

#include <memory>
#include <unordered_map>

namespace csl::nodeeditor {

class CSL_NODE_EDITOR_EXPORT NodeGraph {
 public:
  void addNode(uint32_t id, std::unique_ptr<Node> node);
  void removeNode(uint32_t id);

  void addConnection(
      uint32_t fromID, std::string const& fromSocket, uint32_t toID, std::string const& toSocket);
  void removeConnection(
      uint32_t fromID, std::string const& fromSocket, uint32_t toID, std::string const& toSocket);

  Connection& getOutputConnection(uint32_t fromID, std::string const& fromSocket) const;
  Connection& getInputConnection(uint32_t toID, std::string const& toSocket) const;

 private:
  std::unordered_map<uint32_t, std::unique_ptr<Node>> mNodes;

  std::unordered_map<std::string, std::shared_ptr<Connection>> mInputConnections;
  std::unordered_map<std::string, std::shared_ptr<Connection>> mOutputConnections;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_GRAPH_HPP
