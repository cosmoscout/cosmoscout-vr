////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "NodeGraph.hpp"

#include "Node.hpp"
#include "logger.hpp"

#include <unordered_set>

namespace csl::nodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

NodeGraph::NodeGraph(std::shared_ptr<WebSocket> socket)
    : mSocket(std::move(socket)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

NodeGraph::~NodeGraph() = default;

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

Connection const* NodeGraph::getInputConnection(
    uint32_t toNode, std::string const& toSocket) const {

  auto it = std::find_if(mConnections.begin(), mConnections.end(),
      [&](Connection const& c) { return c.mToNode == toNode && c.mToSocket == toSocket; });

  if (it == mConnections.end()) {
    return nullptr;
  }

  return &(*it);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<Connection const*> NodeGraph::getInputConnections(uint32_t toNode) const {

  std::vector<Connection const*> result;

  for (auto const& c : mConnections) {
    if (c.mToNode == toNode) {
      result.push_back(&c);
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<Connection const*> NodeGraph::getOutputConnections(
    uint32_t fromNode, std::string const& fromSocket) const {

  std::vector<Connection const*> result;

  for (auto const& c : mConnections) {
    if (c.mFromNode == fromNode && c.mFromSocket == fromSocket) {
      result.push_back(&c);
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<Connection const*> NodeGraph::getOutputConnections(uint32_t fromNode) const {

  std::vector<Connection const*> result;

  for (auto const& c : mConnections) {
    if (c.mFromNode == fromNode) {
      result.push_back(&c);
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::process() {
  // logger().debug(
  //     "processing node graph (nodes: {} / connections: {})", mNodes.size(), mConnections.size());

  std::unordered_set<uint32_t> dirtyNodes;

  // First collect all nodes which have changed input sockets.
  for (auto const& c : mConnections) {
    if (c.mHasNewData) {
      dirtyNodes.insert(c.mToNode);
    }
  }

  // All dirty nodes will need to be processed. Processing them will most likely result in changed
  // output values, so we will mark all connectd nodes as being dirty as well. This may result in
  // too many dirty nodes, however, the nodes can check their input connections during the
  // processing and abort early if nothing has changed.
  std::vector<uint32_t> stack(dirtyNodes.begin(), dirtyNodes.end());

  while (!stack.empty()) {
    auto nodes = getOutputNodes(stack.back());
    stack.pop_back();

    dirtyNodes.insert(nodes.begin(), nodes.end());
    stack.insert(stack.end(), nodes.begin(), nodes.end());
  }

  // logger().debug("  dirty nodes: {}", dirtyNodes.size());

  // Now that we have all nodes which should be processed, we process them one by one. We always
  // choose a node which does not receive input from a node which is still dirty.
  while (!dirtyNodes.empty()) {
    auto it = std::find_if(dirtyNodes.begin(), dirtyNodes.end(), [&](uint32_t node) {
      for (uint32_t inputNode : getInputNodes(node)) {
        if (dirtyNodes.find(inputNode) != dirtyNodes.end()) {
          return false;
        }
      }
      return true;
    });

    // logger().debug("    processing node {}", *it);

    mNodes.at(*it)->process();

    dirtyNodes.erase(it);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<uint32_t> NodeGraph::getOutputNodes(uint32_t node) const {
  std::vector<uint32_t> result;

  for (auto const& c : mConnections) {
    if (c.mFromNode == node) {
      result.push_back(c.mToNode);
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<uint32_t> NodeGraph::getInputNodes(uint32_t node) const {
  std::vector<uint32_t> result;

  for (auto const& c : mConnections) {
    if (c.mToNode == node) {
      result.push_back(c.mFromNode);
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::nodeeditor
