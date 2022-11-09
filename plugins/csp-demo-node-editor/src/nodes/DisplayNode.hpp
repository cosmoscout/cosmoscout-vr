////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_DEMO_NODE_EDITOR_DISPLAY_NODE_HPP
#define CSP_DEMO_NODE_EDITOR_DISPLAY_NODE_HPP

#include "../../../csl-node-editor/src/Node.hpp"

namespace csp::demonodeeditor {

///
class DisplayNode : public csl::nodeeditor::Node {
 public:
  static const std::string            NAME;
  static const std::string            SOURCE;
  static std::unique_ptr<DisplayNode> create();

  std::string const& getName() const override;

  void process() override;
};

} // namespace csp::demonodeeditor

#endif // CSP_DEMO_NODE_EDITOR_DISPLAY_NODE_HPP
