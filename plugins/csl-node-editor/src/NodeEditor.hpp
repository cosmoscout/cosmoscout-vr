////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_EDITOR_HPP
#define CSL_NODE_EDITOR_NODE_EDITOR_HPP

#include <memory>
#include <unordered_map>

class CivetServer;
class CivetHandler;

namespace csl::nodeeditor {

class NodeEditor {
 public:
  NodeEditor(uint16_t port);
  ~NodeEditor();

 private:
  void startServer(uint16_t port);
  void quitServer();

  std::unique_ptr<CivetServer>                                   mServer;
  std::unordered_map<std::string, std::unique_ptr<CivetHandler>> mHandlers;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_EDITOR_HPP
