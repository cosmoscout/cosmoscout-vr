////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "NodeGraph.hpp"

#include "../Node.hpp"
#include "../logger.hpp"

namespace csl::nodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

// These need to be declared explicitely as the default versions would be defined inline in the
// header which makes it impossible to use a forward declartion of Node.
NodeGraph::NodeGraph()  = default;
NodeGraph::~NodeGraph() = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::queueProcess() {
  for (auto const& [id, node] : mNodes) {
    mDirtyNodes.insert(id);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::queueProcessing(uint32_t node) {
  mDirtyNodes.insert(node);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json NodeGraph::toJSON() const {
  auto nodes = nlohmann::json::object();

  for (auto const& [id, node] : mNodes) {

    // clang-format off
    nodes[std::to_string(id)] = {
      {"name",      node->getName()},
      {"id",        id},
      {"position",  node->getPosition()},
      {"collapsed", node->getIsCollapsed()},
      {"data",      node->getData()},
      {"outputs",   nlohmann::json::object()}
    };
    // clang-format on
  }

  for (auto const& c : mConnections) {
    nodes[std::to_string(c.mFromNode)]["outputs"][c.mFromSocket]["connections"].push_back(
        {{"node", c.mToNode}, {"input", c.mToSocket}});
  }

  return {{"nodes", nodes}};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::addNode(uint32_t id, std::unique_ptr<Node> node) {
  mNodes.emplace(id, std::move(node));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::removeNode(uint32_t id) {
  mNodes.erase(id);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::setNodePosition(uint32_t id, std::array<int32_t, 2> position) const {
  auto it = mNodes.find(id);

  if (it != mNodes.end()) {
    it->second->setPosition(std::move(position));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::setNodeCollapsed(uint32_t id, bool collapsed) const {
  auto it = mNodes.find(id);

  if (it != mNodes.end()) {
    it->second->setIsCollapsed(collapsed);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::addConnection(
    uint32_t fromNode, std::string fromSocket, uint32_t toNode, std::string toSocket) {

  mDirtyNodes.insert(fromNode);

  mConnections.emplace_back(fromNode, std::move(fromSocket), toNode, std::move(toSocket));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::removeConnection(uint32_t fromNode, std::string const& fromSocket, uint32_t toNode,
    std::string const& toSocket) {

  mDirtyNodes.insert(toNode);

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

  // A naive implementation of this method would simply pop nodes from mDirtyNodes and process them
  // until mDirtyNodes is empty. This would work as new nodes are added to mDirtyNodes if a node
  // produces a new output value during its process() method.
  // However, this approach would lead to too many calls to process(). In the worst case, the
  // process() method of a node would be called once for each connected input. To prevent this, we
  // first collect all nodes which may need to be processed. Then, we process them one by one,
  // always choosing a node for which all input nodes have already been processed.

  // All dirty nodes will need to be processed.
  auto processNodes = mDirtyNodes;

  // Processing them will most likely result in changed output values, so we will process all
  // connected output nodes as well. This could lead to nodes being processed for which the input
  // actually did not change. However, once a node produces a new output, all connected nodes will
  // be put into mDirtyNodes automatically. So we can check if mDirtyNodes contains a specific node
  // before calling process() on it. This way, we can ensure that process() is only called for nodes
  // which have changed input values.

  // Recursively traverse the graph starting from all dirty nodes and push all visited nodes into
  // processNodes. We abort after too many iterations - in this case there must be a cycle in the
  // graph!
  auto     heap       = processNodes;
  uint32_t iterations = 0;

  while (!heap.empty()) {
    auto nodes = getOutputNodes(*heap.begin());
    heap.erase(heap.begin());

    processNodes.insert(nodes.begin(), nodes.end());
    heap.insert(nodes.begin(), nodes.end());

    if (++iterations > mNodes.size() * mNodes.size()) {
      throw std::runtime_error("Cycle detected!");
    }
  }

  // if (!processNodes.empty()) {
  //   logger().debug("dirty: {}, process: {}", mDirtyNodes.size(), processNodes.size());
  // }

  // Now that we have all nodes which should be processed, we process them one by one. We always
  // choose a node which does not receive input from a node which is still pending to be processed.
  while (!processNodes.empty()) {
    auto it = std::find_if(processNodes.begin(), processNodes.end(), [&](uint32_t node) {
      for (uint32_t inputNode : getInputNodes(node)) {
        if (processNodes.find(inputNode) != processNodes.end()) {
          return false;
        }
      }
      return true;
    });

    // Only call process() on nodes which were marked as being dirty.
    if (mDirtyNodes.find(*it) != mDirtyNodes.end()) {
      auto node = mNodes.find(*it);

      if (node != mNodes.end()) {
        // logger().debug(" processing node {}", *it);
        node->second->process();
      }
    } else {
      // logger().debug(" processing of node {} is not required", *it);
    }

    processNodes.erase(it);
  }

  mDirtyNodes.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeGraph::handleNodeMessage(uint32_t toNode, nlohmann::json const& message) const {
  mNodes.at(toNode)->onMessageFromJS(message);
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
