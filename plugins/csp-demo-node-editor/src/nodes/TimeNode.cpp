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

const std::string TimeNode::NAME = "Time";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string TimeNode::SOURCE = R"(
    //js

    // The TimeNode is pretty simple as it only has a single output socket. The component serves as
    // a kind of factory. Whenever a new node is created, the builder() method is called.
    class TimeComponent extends Rete.Component {

      constructor() {
        // This name must match the TimeNode::NAME defined above.
        super("Time");

        // This specifies the submenu from which this node can be created in the node editor.
        this.category = "Inputs";
      }

      // Called whenever a new node of this type needs to be constructed.
      builder(node) {

        // This node has a single output. The first parameter is the name of this output and must be
        // unique amongst all sockets. It is also used in the TimeNode::process() to write the
        // output of this node. The second parameter is shown as name on the node. The last
        // parameter references a socket type which has been registered with the node factory
        // before.
        let output = new Rete.Output('time', "Seconds", CosmoScout.socketTypes['Number Value']);
        node.addOutput(output)

        return node;
      }
    }
    //!js
  )";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<TimeNode> TimeNode::create(std::shared_ptr<cs::core::TimeControl> pTimeControl) {
  return std::make_unique<TimeNode>(pTimeControl);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeNode::TimeNode(std::shared_ptr<cs::core::TimeControl> pTimeControl)
    : mTimeControl(std::move(pTimeControl)) {

  // Whenever the simulation time changes, we write it to the output by calling the process()
  // method. Writing the output will not trigger a graph reprocessing right away, it will only queue
  // up the connected nodes for being processed in the next update step.
  mTimeConnection = mTimeControl->pSimulationTime.connect([this](double) { process(); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeNode::~TimeNode() {
  mTimeControl->pSimulationTime.disconnect(mTimeConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& TimeNode::getName() const {
  return NAME;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeNode::process() {

  // The name of the port must match the name given in the JavaScript code above.
  writeOutput("time", mTimeControl->pSimulationTime.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
