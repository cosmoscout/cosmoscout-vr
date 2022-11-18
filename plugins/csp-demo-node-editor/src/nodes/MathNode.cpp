////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "MathNode.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"

namespace csp::demonodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string MathNode::sName = "Math";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string MathNode::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-demo-node-editor/MathNode.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<MathNode> MathNode::sCreate() {
  return std::make_unique<MathNode>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& MathNode::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MathNode::process() {

  // Whenever this method is called, we compute the output value based on the given input values and
  // the currently selected math operation.
  double first  = readInput<double>("first", 0.0);
  double second = readInput<double>("second", 0.0);

  double result = 0.0;

  switch (mOperation) {
  case Operation::eAdd:
    result = first + second;
    break;
  case Operation::eSubtract:
    result = first - second;
    break;
  case Operation::eMultiply:
    result = first * second;
    break;
  case Operation::eDivide:
    result = first / second;
    break;
  default:
    break;
  }

  writeOutput("result", result);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MathNode::onMessageFromJS(nlohmann::json const& message) {

  // The CosmoScout.sendMessageToCPP() method sends the currently selected math operation.
  mOperation = message;

  // Whenever the operation changes, we write the new output by calling the process() method.
  // Writing the output will not trigger a graph reprocessing right away, it will only queue
  // up the connected nodes for being processed in the next update step.
  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json MathNode::getData() const {
  return {{"operation", mOperation}};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MathNode::setData(nlohmann::json const& json) {
  mOperation = json["operation"];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
