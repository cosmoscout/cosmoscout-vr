////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_DEMO_NODE_EDITOR_NUMBER_NODE_HPP
#define CSP_DEMO_NODE_EDITOR_NUMBER_NODE_HPP

#include "../../../csl-node-editor/src/NodeEditor.hpp"

namespace csp::demonodeeditor {

///
class NumberNode : public csl::nodeeditor::Node {
 public:
  static std::string getName();
  static std::string getSource();

  static std::unique_ptr<NumberNode> create();
};

} // namespace csp::demonodeeditor

#endif // CSP_DEMO_NODE_EDITOR_NUMBER_NODE_HPP
