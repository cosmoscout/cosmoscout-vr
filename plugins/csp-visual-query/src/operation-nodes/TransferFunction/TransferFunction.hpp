////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_VOLUME_TRANSFER_FUNCTION_HPP
#define CSP_VISUAL_QUERY_VOLUME_TRANSFER_FUNCTION_HPP

#include "../../../../csl-node-editor/src/Node.hpp"

#include <memory>

namespace csp::visualquery {

/// Your node description here!
class TransferFunction final : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string               sName;
  static std::string                     sSource();
  static std::unique_ptr<TransferFunction> sCreate();

  // instance interface ----------------------------------------------------------------------------

  /// Each node must override this. It simply returns the static sName.
  std::string const& getName() const override;

  /// Whenever the value of the node is required, the VolumeTransferFunctionNode will send a message to
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
  nlohmann::json getData() const override;

  /// This is called whenever the node needs to be deserialized.
  void setData(nlohmann::json const& json) override;

private:
  std::vector<glm::vec4> mLut;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_VOLUME_TRANSFER_FUNCTION_HPP
