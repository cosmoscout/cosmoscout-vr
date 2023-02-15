////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_FACTORY_HPP
#define CSL_NODE_EDITOR_NODE_FACTORY_HPP

#include "csl_node_editor_export.hpp"

#include "Node.hpp"

namespace csl::nodeeditor {

/// This class is used to register all socket and node types which should be available in a node
/// editor instance. The node editor will use an instance of this class to instantiate nodes which
/// have been registered before. You can have a look at the csp-demo-node-editor plugin for an usage
/// example.
class CSL_NODE_EDITOR_EXPORT NodeFactory {
 public:
  // public API ------------------------------------------------------------------------------------

  // As a user of this library, you will usually only have to call the two methods below.

  /// Registers a new node socket which can be used by nodes of the node editor.
  /// @param name     The unique name of the socket type. This is used to retrieve references to
  ///                 this socket type in the JavaScript part of your nodes. You can access the
  ///                 socket in JavaScript using this: CosmoScout.socketTypes['Socket Name'].
  /// @param color    A string defining the color of the socket. This can be anything which is
  ///                 accepted by CSS. For example 'red', '#f00', or 'rgb(255, 0, 0)'.
  void registerSocketType(std::string name, std::string color);

  /// Registers a new node type which can then be used in the node editor.
  /// @tparam T       The node type to be registered. This should be derived from nodeeditor::Node.
  ///                 See the documentation of this class for more information.
  /// @param ...args  The given arguments will be passed to the static sCreate() method whenever a
  ///                 new node of this type is constructed.
  template <typename T, typename... Args>
  void registerNodeType(Args... args) {
    mNodeSourceFuncs.push_back([=]() { return T::sSource(); });
    mNodeCreateFuncs[T::sName] = [=]() { return T::sCreate(args...); };
  }

  // Node Editor API -------------------------------------------------------------------------------

  // The methods below are primarily meant to be used by the NodeEditor class. You may want use them
  // for debugging purposes, though.

  /// This creates the JavaScript source snippet which is injected into the node editor web page to
  /// setup all registered socket types.
  /// @return The JavaScript source code.
  std::string getSocketSource() const;

  /// This creates the JavaScript source snippet which is injected into the node editor web page to
  /// setup all registered node types.
  /// @return The JavaScript source code.
  std::string getNodeSource() const;

  /// This creates the JavaScript source snippet which is injected into the node editor web page to
  /// register all node types.
  /// @return The JavaScript source code.
  std::string getRegisterSource() const;

  /// Create a new Node given the name of the node type.
  /// @param type The name of the node type (the static NAME of the registered node type).
  /// @return     A newly created node of the given type name.
  /// @throws     This may throw a std::runtime_error if the given type has not been registered
  ///             before.
  std::unique_ptr<Node> createNode(std::string const& type) const;

 private:
  // Stores socket color for each unique socket name.
  std::unordered_map<std::string, std::string> mSockets;

  // Functions to retrieve the JavaScript source of each registered node.
  std::vector<std::function<std::string(void)>> mNodeSourceFuncs;

  // Functions to create new nodes for each type name.
  std::unordered_map<std::string, std::function<std::unique_ptr<Node>(void)>> mNodeCreateFuncs;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_FACTORY_HPP
