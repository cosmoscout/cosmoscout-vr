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
    uint32_t fromID, std::string const& fromSocket, uint32_t toID, std::string const& toSocket) {

  auto connection = std::make_shared<Connection>();

  mInputConnections[fmt::format("{}:{}", toID, toSocket)]      = connection;
  mOutputConnections[fmt::format("{}:{}", fromID, fromSocket)] = connection;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void NodeGraph::removeConnection(
    uint32_t fromID, std::string const& fromSocket, uint32_t toID, std::string const& toSocket) {

  mInputConnections.erase(fmt::format("{}:{}", toID, toSocket));
  mOutputConnections.erase(fmt::format("{}:{}", fromID, fromSocket));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Connection& NodeGraph::getOutputConnection(uint32_t fromID, std::string const& fromSocket) const {
  return *mOutputConnections.at(fmt::format("{}:{}", fromID, fromSocket)).get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Connection& NodeGraph::getInputConnection(uint32_t toID, std::string const& toSocket) const {
  return *mInputConnections.at(fmt::format("{}:{}", toID, toSocket)).get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::nodeeditor
