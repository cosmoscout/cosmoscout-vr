////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_DEMO_NODE_EDITOR_MATH_NODE_HPP
#define CSP_DEMO_NODE_EDITOR_MATH_NODE_HPP

#include "../../../csl-node-editor/src/Node.hpp"

namespace csp::demonodeeditor {

/// The MathNode is the most complex node of this node editor example. It has two inputs, a
/// user-selectable math operation and an output. It demonstrates how a node can work with input
/// socket data and user-provided data. The node will write a new value to its output whenever the
/// user edits the math operation or whenever an input value is changed. The current math operation
/// is stored as a private member and will be serialized and deserialized whenever the node graph is
/// saved or loaded.
class MathNode : public csl::nodeeditor::Node {
 public:
  /// These operations are supported by the MathNode.
  enum class Operation { eAdd, eSubtract, eMultiply, eDivide };

  // static interface ------------------------------------------------------------------------------

  static const std::string         sName;
  static std::string               sSource();
  static std::unique_ptr<MathNode> sCreate();

  // instance interface ----------------------------------------------------------------------------

  /// Each node must override this. It simply returns the static sName.
  std::string const& getName() const override;

  /// Whenever the user changes the math operation, the MathNode will send a message to the C++
  /// instance of the node via onMessageFromJS, which in turn will call the process() method. This
  /// will read the input socket values and compute and write the output socket value accordingly.
  /// This method may also get called occasionally by the node editor, for example if a new web
  /// client was connected hence needs updated values for all nodes.
  void process() override;

  /// This will be called whenever the CosmoScout.sendMessageToCPP() is called by the JavaScript
  /// client part of this node.
  /// @param message  A JSON object as sent by the JavaScript node. In this case, it is actually
  ///                 just the currently selected math operation.
  void onMessageFromJS(nlohmann::json const& message) override;

  /// This is called whenever the node needs to be serialized. It returns a JSON object containing
  /// the currently selected math operation (as number).
  nlohmann::json getData() const override;

  /// This is called whenever the node needs to be deserialized. The given JSON object should
  /// contain a number corresponding to a math operation under the key "operation".
  void setData(nlohmann::json const& json) override;

 private:
  Operation mOperation = Operation::eAdd;
};

} // namespace csp::demonodeeditor

#endif // CSP_DEMO_NODE_EDITOR_MATH_NODE_HPP
