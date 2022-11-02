////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "DisplayNode.hpp"

#include "../logger.hpp"

#include "../../../../src/cs-utils/utils.hpp"

namespace csp::demonodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string DisplayNode::getName() {
  return "Display";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string DisplayNode::getSource() {
  std::string source = R"(
    class %NAME%Control extends Rete.Control {

      constructor(key) {
        super(key);

        this.component = {
          template: '<div>Huhu</div>',
        };
      }
    }

    class %NAME%Component extends Rete.Component {

      constructor() {
        super("%NAME%");

        this.category = "Outputs";
      }

      builder(node) {
        let input = new Rete.Input('number', "Number", CosmoScout.socketTypes['Number Value']);
        return node.addControl(new %NAME%Control('num')).addInput(input);
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

void DisplayNode::process() {

  if (hasNewInput()) {
    double value = readInput<double>("number", 0.0);
    logger().info("{} got {}", mID, value);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
