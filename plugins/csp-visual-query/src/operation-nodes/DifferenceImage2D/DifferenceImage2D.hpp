////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_DIFFERENCEIMAGE2D_NODE_HPP
#define CSP_VISUAL_QUERY_DIFFERENCEIMAGE2D_NODE_HPP

#include "../../../../csl-node-editor/src/Node.hpp"
#include "../../types/types.hpp"

namespace csp::visualquery {

/// The DifferenceImage2DNode calculates the difference between the first and second input.
/// The node will write a new value to its output whenever a reprocessing of the connected nodes is
/// triggered. The current value is stored as a private member and will be serialized and
/// deserialized whenever the node graph is saved or loaded.
class DifferenceImage2D : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string                  sName;
  static std::string                        sSource();
  static std::unique_ptr<DifferenceImage2D> sCreate();

  // instance interface ----------------------------------------------------------------------------

  /// Each node must override this. It simply returns the static sName.
  std::string const& getName() const override;

  /// Whenever the value of the node is required, the DifferenceImage2DNode will send a message to
  /// the C++ instance of the node via onMessageFromJS, which in turn will call the process()
  /// method.
  /// This method may also get called occasionally by the node editor,
  /// for example if a new web client was connected hence needs updated values for all nodes.
  void process() override;

  /// This will be called whenever the CosmoScout.sendMessageToCPP() is called by the JavaScript
  /// client part of this node.
  /// @param message  A JSON object as sent by the JavaScript node.
  void onMessageFromJS(nlohmann::json const& message) override;

  /// This is called whenever the node needs to be serialized.
  /// It returns a JSON object containing the current value.
  nlohmann::json getData() const override;

  /// This is called whenever the node needs to be deserialized.
  /// The given JSON object should contain a number under the key "value". the current value.
  void setData(nlohmann::json const& json) override;

 private:
  std::shared_ptr<Image2D> mValue;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_DIFFERENCEIMAGE2D_NODE_HPP