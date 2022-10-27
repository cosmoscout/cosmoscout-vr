////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_FACTORY_HPP
#define CSL_NODE_EDITOR_NODE_FACTORY_HPP

#include "Node.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace csl::nodeeditor {

/// This class is used to register all socket and node types which should be available in a node
/// editor instance. The node editor will use an instance of this class to instantiate nodes which
/// have been registered before. You can have a look at the csp-demo-node-editor plugin for an usage
/// example.
class CSL_NODE_EDITOR_EXPORT NodeFactory {
 public:
  /// Registers a new node socket which can be used by nodes of the node editor.
  ///
  /// @param name          The unique name of the socket type. This will be used for the tooltip
  ///                      when hovering over a socket of this type. It is also used to retrieve
  ///                      references to this socket type in the JavaScript part of your nodes.
  ///                      You can access the socket in JavaScript using this:
  ///                      CosmoScout.socketTypes['Socket Name'].
  /// @param color         A string defining the color of the socket. This can be anything which is
  ///                      accepted by CSS. For example 'red', '#f00', or 'rgb(255, 0, 0)'.
  /// @param compatibleTo  A list of other socket types which allow this socket type to be connected
  ///                      to (not vice-versa).
  void registerSocketType(
      std::string const& name, std::string color, std::vector<std::string> compatibleTo = {});

  /// Registers a new node type which can then be used in the node editor.
  ///
  /// @tparam T       The node type to be registered. This should be derived
  ///                 from csl::nodeeditor::Node and implement at least these three static methods:
  ///                  * static std::string getName();
  ///                  * static std::string getSource();
  ///                  * static std::unique_ptr<T> create(args...);
  /// @param ...args  The given arguments will be passed to the static create method whenever a new
  ///                 node of this type is constructed.
  template <typename T, typename... Args>
  void registerNodeType(Args... args) {
    mNodeSourceFuncs.push_back([=]() { return T::getSource(); });
    mNodeCreateFuncs[T::getName()] = [=]() { return T::create(args...); };
  }

  // Node Editor API -------------------------------------------------------------------------------

  // The methods below are primarily meant to be used by the NodeEditor class. You may want use them
  // for debugging purposes, though.

  /// This creates the JavaScript source snippet which is injected into the node editor to setup all
  /// registered socket types.
  /// @return The JavaScript source code.
  std::string getSocketSource() const;

  /// This creates the JavaScript source snippet which is injected into the node editor to setup all
  /// registered node types.
  /// @return The JavaScript source code.
  std::string getNodeSource() const;

  /// This creates the JavaScript source snippet which is injected into the node editor to register
  /// all node types.
  /// @return The JavaScript source code.
  std::string getRegisterSource() const;

  /// Create a new Node given the name of the node type.
  /// @param type The name of the node type (as returned by the static getName() method of the
  ///             registered node type).
  /// @return     A newly created node of the given type name.
  /// @throws     This may throw a std::runtime_error is the given type has not been registered
  ///             before.
  std::unique_ptr<Node> createNode(std::string const& type) const;

 private:
  struct SocketInfo {
    std::string              mColor;
    std::vector<std::string> mCompatibleTo;
  };

  // Stores socket information for each unique socket name.
  std::unordered_map<std::string, SocketInfo> mSockets;

  // Functions to retrieve the JavaScript source of each registered node.
  std::vector<std::function<std::string(void)>> mNodeSourceFuncs;

  // Functions to create new nodes for each type name.
  std::unordered_map<std::string, std::function<std::unique_ptr<Node>(void)>> mNodeCreateFuncs;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_FACTORY_HPP
