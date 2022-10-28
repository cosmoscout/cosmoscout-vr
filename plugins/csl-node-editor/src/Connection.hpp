////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_CONNECTION_HPP
#define CSL_NODE_EDITOR_CONNECTION_HPP

#include "csl_node_editor_export.hpp"

#include <any>
#include <string>

namespace csl::nodeeditor {

struct CSL_NODE_EDITOR_EXPORT Connection {
  Connection(uint32_t fromNode, std::string fromSocket, uint32_t toNode, std::string toSocket)
      : mFromNode(fromNode)
      , mFromSocket(std::move(fromSocket))
      , mToNode(toNode)
      , mToSocket(std::move(toSocket)) {
  }

  uint32_t    mFromNode;
  std::string mFromSocket;

  uint32_t    mToNode;
  std::string mToSocket;

  mutable std::any mValue;
  mutable bool     mHasNewData = false;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_CONNECTION_HPP
