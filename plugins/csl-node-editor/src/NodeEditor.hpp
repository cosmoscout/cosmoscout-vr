////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_EDITOR_HPP
#define CSL_NODE_EDITOR_NODE_EDITOR_HPP

#include "NodeFactory.hpp"

#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

class CivetServer;
class CivetHandler;
class CivetWebSocketHandler;

namespace csl::nodeeditor {

class NodeGraph;
class WebSocket;

class CSL_NODE_EDITOR_EXPORT NodeEditor {
 public:
  NodeEditor(uint16_t port, NodeFactory factory);
  ~NodeEditor();

  void update();

  nlohmann::json toJSON() const;
  void           fromJSON(nlohmann::json const& json);

 private:
  void startServer(uint16_t port);
  void quitServer();

  void handleAddNodeEvent(nlohmann::json const& json);
  void handleRemoveNodeEvent(nlohmann::json const& json);
  void handleTranslateNodeEvent(nlohmann::json const& json);
  void handleCollapseNodeEvent(nlohmann::json const& json);
  void handleAddConnectionEvent(nlohmann::json const& json);
  void handleRemoveConnectionEvent(nlohmann::json const& json);
  void handleNodeMessageEvent(nlohmann::json const& json);

  std::string createHTMLSource() const;

  NodeFactory                mFactory;
  std::shared_ptr<WebSocket> mSocket;
  std::shared_ptr<NodeGraph> mGraph;

  std::unique_ptr<CivetServer>                                       mServer;
  std::vector<std::pair<std::string, std::unique_ptr<CivetHandler>>> mHandlers;

  std::string mHTMLSource;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_EDITOR_HPP
