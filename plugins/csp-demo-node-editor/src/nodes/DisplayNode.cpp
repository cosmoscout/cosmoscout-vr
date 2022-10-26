////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "DisplayNode.hpp"

#include "../../../../src/cs-utils/utils.hpp"

namespace csp::demonodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string DisplayNode::getName() {
  return "Display";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string DisplayNode::getSource() {
  std::string source = R"(
    class %NAME%Component extends Rete.Component {

      constructor() {
        super("%NAME%");

        this.category = "Outputs";
      }

      builder(node) {
        let input = new Rete.Input('number', "Number", CosmoScout.socketTypes['Number Value']);
        return node.addInput(input);
      }

      worker(node, inputs, outputs) {
        let val = inputs['number'].length ? inputs['number'][0] : NaN;
        CosmoScout.connection.send("process");
      }
    }
  )";

  cs::utils::replaceString(source, "%NAME%", getName());

  return source;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<DisplayNode> DisplayNode::create() {
  return std::make_unique<DisplayNode>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
