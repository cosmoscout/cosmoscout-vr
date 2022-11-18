////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TimeNode.hpp"

#include "../../../../src/cs-core/TimeControl.hpp"
#include "../../../../src/cs-utils/filesystem.hpp"

namespace csp::demonodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string TimeNode::sName = "Time";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string TimeNode::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-demo-node-editor/TimeNode.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<TimeNode> TimeNode::sCreate(std::shared_ptr<cs::core::TimeControl> pTimeControl) {
  return std::make_unique<TimeNode>(std::move(pTimeControl));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeNode::TimeNode(std::shared_ptr<cs::core::TimeControl> timeControl)
    : mTimeControl(std::move(timeControl)) {

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
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeNode::process() {

  // The name of the port must match the name given in the JavaScript code above.
  writeOutput("time", mTimeControl->pSimulationTime.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
