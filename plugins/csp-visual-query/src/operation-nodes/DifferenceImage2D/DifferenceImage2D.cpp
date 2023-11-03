////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "DifferenceImage2D.hpp"
#include "../../../../src/cs-utils/utils.hpp"
#include "../../logger.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string DifferenceImage2D::sName = "DifferenceImage2D";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string DifferenceImage2D::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/DifferenceImage2D.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<DifferenceImage2D> DifferenceImage2D::sCreate() {
  return std::make_unique<DifferenceImage2D>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& DifferenceImage2D::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
std::vector<std::vector<T>> getDifference(std::vector<std::vector<T>> first, std::vector<std::vector<T>> second) {
  // Init Result
  std::vector<std::vector<T>> result;
  // Resize result vector to input size
  result.resize(first.size());

  // Loop over available points
  for (int iPoint = 0; iPoint < first.size(); iPoint++) {
    // Resize point's vector (scalars) to size of input points
    result[iPoint].resize(first[iPoint].size());

    // Loop over vector of each point (scalars)
    for (int iScalar = 0; iScalar < first[iPoint].size(); iScalar++) {
      result[iPoint][iScalar] = first[iPoint][iScalar] - second[iPoint][iScalar];
    }
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DifferenceImage2D::process() {
  // Read inputs
  auto first  = readInput<std::shared_ptr<Image2D>>("first", nullptr);
  auto second = readInput<std::shared_ptr<Image2D>>("second", nullptr);
  // Check exists
  if (!first || !second) {
    return;
  }
  // Check valid
  nlohmann::json message;
  message["status"] = "OK";
  if (first->mDimension != second->mDimension) {
    message["status"] = "ERROR";
    message["error"].push_back("DimensionMismatch");
  }
  if (first->mBounds != second->mBounds) {
    message["status"] = "ERROR";
    message["error"].push_back("BoundsMismatch");
  }
  if (first->mTimeStamp != second->mTimeStamp) {
    message["status"] = "ERROR";
    message["error"].push_back("TimestampMismatch");
  }
  if (first->mNumScalars != second->mNumScalars) {
    message["status"] = "ERROR";
    message["error"].push_back("NumScalarsMismatch");
  }
  if (first->mPoints.index() != second->mPoints.index()) {
    message["status"] = "ERROR";
    message["error"].push_back("PointsTypeMismatch");
  }
  sendMessageToJS(message);
  if (message["status"] == "ERROR") {
    return;
  }

  // Init return value
  PointsType diffPoints;

  // Check wich type is held by variant to calculate differnce
  if (std::holds_alternative<U8ValueVector>(first->mPoints) && std::holds_alternative<U8ValueVector>(second->mPoints)) {
    diffPoints = getDifference(std::get<U8ValueVector>(first->mPoints), std::get<U8ValueVector>(second->mPoints));
  }
  else if (std::holds_alternative<U16ValueVector>(first->mPoints) && std::holds_alternative<U16ValueVector>(second->mPoints)) {
    diffPoints = getDifference(std::get<U16ValueVector>(first->mPoints), std::get<U16ValueVector>(second->mPoints));
  }
  else if (std::holds_alternative<U32ValueVector>(first->mPoints) && std::holds_alternative<U32ValueVector>(second->mPoints)) {
    diffPoints = getDifference(std::get<U32ValueVector>(first->mPoints), std::get<U32ValueVector>(second->mPoints));
  }
  else if (std::holds_alternative<I16ValueVector>(first->mPoints) && std::holds_alternative<I16ValueVector>(second->mPoints)) {
    diffPoints = getDifference(std::get<I16ValueVector>(first->mPoints), std::get<I16ValueVector>(second->mPoints));
  }
  else if (std::holds_alternative<I32ValueVector>(first->mPoints) && std::holds_alternative<I32ValueVector>(second->mPoints)) {
    diffPoints = getDifference(std::get<I32ValueVector>(first->mPoints), std::get<I32ValueVector>(second->mPoints));
  }
  else if (std::holds_alternative<F32ValueVector>(first->mPoints) && std::holds_alternative<F32ValueVector>(second->mPoints)) {
    diffPoints = getDifference(std::get<F32ValueVector>(first->mPoints), std::get<F32ValueVector>(second->mPoints));
  }
  else {
    logger().error("Unknown type in input variants!");
  }

  // Fill output value (reuse first socket metadata)
  mValue = std::make_shared<Image2D>(
      diffPoints, first->mNumScalars, first->mDimension, first->mBounds, first->mTimeStamp);

  writeOutput("value", mValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DifferenceImage2D::onMessageFromJS(nlohmann::json const& message) {
  /*
  // Processing the node will not trigger a graph reprocessing right away, it will only queue
  // up the connected nodes for being processed in the next update step (and only if the value
  // actually changed).
  process();
  */
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json DifferenceImage2D::getData() const {
  // Nothing to serialize
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DifferenceImage2D::setData(nlohmann::json const& json) {
  // Nothing to deserialize
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
