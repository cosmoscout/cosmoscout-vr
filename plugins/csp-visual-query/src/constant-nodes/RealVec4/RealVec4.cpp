////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "RealVec4.hpp"

#include "../../../../../src/cs-utils/filesystem.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string RealVec4::sName = "RealVec4";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string RealVec4::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/RealVec4.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<RealVec4> RealVec4::sCreate() {
  return std::make_unique<RealVec4>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& RealVec4::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RealVec4::process() {
  writeOutput("value", mValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RealVec4::onMessageFromJS(nlohmann::json const& message) {
  // The message sent via CosmoScout.sendMessageToCPP() contains the selected number.
  if (message.contains("real1")) {
    mValue[0] = message["real1"];
  } else if (message.contains("real2")) {
    mValue[1] = message["real2"];
  } else if (message.contains("real3")) {
    mValue[2] = message["real3"];
  } else {
    mValue[3] = message["real4"];
  }

  // Whenever the user entered a number, we write it to the output socket by calling the process()
  // method. Writing the output will not trigger a graph reprocessing right away, it will only queue
  // up the connected nodes for being processed in the next update step (and only if the value
  // actually changed).
  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json RealVec4::getData() const {
  return {{"value", mValue}};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RealVec4::setData(nlohmann::json const& json) {
  mValue = json["value"];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
