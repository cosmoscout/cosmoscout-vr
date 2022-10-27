////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_EDITOR_HPP
#define CSL_NODE_EDITOR_NODE_EDITOR_HPP

#include "NodeFactory.hpp"

#include <memory>
#include <vector>

class CivetServer;
class CivetHandler;
class CivetWebSocketHandler;

namespace csl::nodeeditor {

class CSL_NODE_EDITOR_EXPORT NodeEditor {
 public:
  NodeEditor(uint16_t port, NodeFactory factory);
  ~NodeEditor();

  void update() const;

 private:
  void startServer(uint16_t port);
  void quitServer();

  std::string createHTMLSource() const;

  NodeFactory mFactory;

  std::unique_ptr<CivetServer>                                       mServer;
  std::vector<std::pair<std::string, std::unique_ptr<CivetHandler>>> mHandlers;
  std::unique_ptr<CivetWebSocketHandler>                             mSocket;

  std::string mHTMLSource;

  std::unordered_map<std::string, std::unique_ptr<Node>> mNodes;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_EDITOR_HPP
