////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_HPP
#define CSL_NODE_EDITOR_NODE_HPP

#include "csl_node_editor_export.hpp"

#include "internal/NodeGraph.hpp"

#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace csl::nodeeditor {

class CommunicationChannel;

class CSL_NODE_EDITOR_EXPORT Node {
 public:
  Node()          = default;
  virtual ~Node() = default;

  /// This will be called whenever the values of one or multiple input sockets have changed.
  virtual void process() = 0;

  void     setID(uint32_t id);
  uint32_t getID() const;

  void                          setPosition(std::array<int32_t, 2> position);
  std::array<int32_t, 2> const& getPosition() const;

  /// @brief
  /// @param collapsed
  void setIsCollapsed(bool collapsed);

  bool getIsCollapsed() const;

  void setSocket(std::shared_ptr<CommunicationChannel> socket);
  void setGraph(std::shared_ptr<NodeGraph> graph);

  virtual std::string const& getName() const = 0;

  void         sendMessageToJS(nlohmann::json const& message) const;
  virtual void onMessageFromJS(nlohmann::json const& message){};

  virtual nlohmann::json getData() const {
    return nlohmann::json::object();
  };

  virtual void setData(nlohmann::json const& json){};

 protected:
  ///
  /// @tparam T
  /// @param socket
  /// @param value
  template <typename T>
  void writeOutput(std::string const& socket, T const& value) {
    auto connections = mGraph->getOutputConnections(mID, socket);

    for (auto& c : connections) {
      if (!c->mData.has_value() || std::any_cast<T>(c->mData) != value) {
        mGraph->queueProcessing(c->mToNode);
        c->mData = value;
      }
    }
  }

  ///
  /// @tparam T
  /// @param socket
  /// @param defaultValue
  /// @return
  template <typename T>
  T readInput(std::string const& socket, T defaultValue) {
    auto connection = mGraph->getInputConnection(mID, socket);

    if (connection && connection->mData.has_value()) {
      return std::any_cast<T>(connection->mData);
    }

    return std::move(defaultValue);
  }

  uint32_t                              mID = 0;
  std::array<int32_t, 2>                mPosition;
  bool                                  mIsCollapsed = false;
  std::shared_ptr<CommunicationChannel> mSocket;
  std::shared_ptr<NodeGraph>            mGraph;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_HPP
