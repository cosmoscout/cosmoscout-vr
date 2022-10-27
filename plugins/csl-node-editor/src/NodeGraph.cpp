////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "NodeGraph.hpp"

#include "logger.hpp"

namespace csl::nodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::addNode(uint32_t id, std::unique_ptr<Node> node) {
  mNodes.emplace(id, std::move(node));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::removeNode(uint32_t id) {
  mNodes.erase(id);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::addConnection(
    uint32_t fromNode, std::string fromSocket, uint32_t toNode, std::string toSocket) {

  mConnections.emplace_back(fromNode, std::move(fromSocket), toNode, std::move(toSocket));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::removeConnection(uint32_t fromNode, std::string const& fromSocket, uint32_t toNode,
    std::string const& toSocket) {

  mConnections.remove_if([&](Connection const& c) {
    return c.mFromNode == fromNode && c.mFromSocket == fromSocket && c.mToNode == toNode &&
           c.mToSocket == toSocket;
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<Connection const&> NodeGraph::getInputConnection(
    uint32_t toNode, std::string const& toSocket) const {

  auto it = std::find_if(mConnections.begin(), mConnections.end(),
      [&](Connection const& c) { return c.mToNode == toNode && c.mToSocket == toSocket; });

  if (it == mConnections.end()) {
    return std::nullopt;
  }

  return *it;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<Connection const&> NodeGraph::getOutputConnections(
    uint32_t fromNode, std::string const& fromSocket) const {

  std::vector<Connection const&> result;

  for (auto const& c : mConnections) {
    if (c.mFromNode == fromNode && c.mFromSocket == fromSocket) {
      result.push_back(c);
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::nodeeditor
