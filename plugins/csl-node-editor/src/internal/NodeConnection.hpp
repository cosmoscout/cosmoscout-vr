////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_CONNECTION_HPP
#define CSL_NODE_EDITOR_NODE_CONNECTION_HPP

#include <any>
#include <string>

namespace csl::nodeeditor {

/// A NodeConnection is the C++ counterpart of the wiggly line connecting an output socket to an
/// input socket of two nodes. It is used for transmitting data from one node to the other by means
/// of an std::any.
struct NodeConnection {
  NodeConnection(uint32_t fromNode, std::string fromSocket, uint32_t toNode, std::string toSocket)
      : mFromNode(fromNode)
      , mFromSocket(std::move(fromSocket))
      , mToNode(toNode)
      , mToSocket(std::move(toSocket)) {
  }

  uint32_t    mFromNode;
  std::string mFromSocket;

  uint32_t    mToNode;
  std::string mToSocket;

  mutable std::any mData;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_CONNECTION_HPP
