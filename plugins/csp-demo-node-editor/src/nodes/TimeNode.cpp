////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TimeNode.hpp"

#include "../logger.hpp"

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
        let output = new Rete.Output('date', "Date", CosmoScout.socketTypes['Date Value']);
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
  return std::make_unique<TimeNode>(pTimeControl);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeNode::TimeNode(std::shared_ptr<cs::core::TimeControl> pTimeControl)
    : mTimeControl(std::move(pTimeControl)) {

  mTimeConnection = mTimeControl->pSimulationTime.connect([this](double value) {
    auto connection = getOutputConnection("date");
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeNode::~TimeNode() {
  mTimeControl->pSimulationTime.disconnect(mTimeConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
