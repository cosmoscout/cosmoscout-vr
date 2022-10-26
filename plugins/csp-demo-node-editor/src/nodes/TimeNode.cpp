////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TimeNode.hpp"

#include "../../../../src/cs-core/TimeControl.hpp"
#include "../../../../src/cs-utils/utils.hpp"

namespace csp::demonodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string TimeNode::getName() {
  return "Time";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string TimeNode::getSource() {
  std::string source = R"(
    class %NAME%Component extends Rete.Component {

      constructor() {
        super("%NAME%");

        this.category = "Inputs";
      }

      builder(node) {
        let output = new Rete.Output('date', "Date", SOCKETS['Date Value']);
        return node.addOutput(output);
      }

      worker(node, inputs, outputs) {
        outputs['date'] = 42;
      }
    }
  )";

  cs::utils::replaceString(source, "%NAME%", getName());

  return source;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<TimeNode> TimeNode::create(std::shared_ptr<cs::core::TimeControl> pTimeControl) {
  return std::make_unique<TimeNode>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
