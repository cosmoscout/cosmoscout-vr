////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_DEMO_NODE_EDITOR_DISPLAY_NODE_HPP
#define CSP_DEMO_NODE_EDITOR_DISPLAY_NODE_HPP

#include "../../../csl-node-editor/src/Node.hpp"

namespace csp::demonodeeditor {

/// This node simply displays the value which is given to its input socket. As the data is passed
/// only between the C++ nodes of node editor server, this requires sending the current input value
/// to the JavaScript node of the node editor client whenever the process() method is called.
class DisplayNode : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string            NAME;
  static const std::string            SOURCE;
  static std::unique_ptr<DisplayNode> create();

  // instance interface ----------------------------------------------------------------------------

  /// Each node must override this. It simply returns the static NAME.
  std::string const& getName() const override;

  /// This is called by the node editor whenever the displayed value needs to be redrawn. This could
  /// be when the input connection has new data available or whenever a new client connected to the
  /// node editor.
  void process() override;
};

} // namespace csp::demonodeeditor

#endif // CSP_DEMO_NODE_EDITOR_DISPLAY_NODE_HPP
