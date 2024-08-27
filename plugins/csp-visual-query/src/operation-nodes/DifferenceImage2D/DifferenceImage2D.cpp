////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "DifferenceImage2D.hpp"
#include "../../../../src/cs-utils/filesystem.hpp"
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

template <typename T>
PointsType getDifference(PointsType const& first, PointsType const& second) {
  T const& v1 = std::get<T>(first);
  T const& v2 = std::get<T>(second);
  // Init Result
  T result;
  // Resize result vector to input size
  result.resize(v1.size());

  // Loop over available points
  for (size_t iPoint = 0; iPoint < v1.size(); iPoint++) {
    // Resize point's vector (scalars) to size of input points
    result[iPoint].resize(v1[iPoint].size());

    // Loop over vector of each point (scalars)
    for (size_t iScalar = 0; iScalar < v1[iPoint].size(); iScalar++) {
      result[iPoint][iScalar] = v1[iPoint][iScalar] - v2[iPoint][iScalar];
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

  switch (first->mPoints.index()) {
  case cs::utils::variantIndex<PointsType, U8ValueVector>():
    logger().info("U8ValueVector found!");
    diffPoints = getDifference<U8ValueVector>(first->mPoints, second->mPoints);
    break;
  case cs::utils::variantIndex<PointsType, U16ValueVector>():
    logger().info("U16ValueVector found!");
    diffPoints = getDifference<U16ValueVector>(first->mPoints, second->mPoints);
    break;
  case cs::utils::variantIndex<PointsType, U32ValueVector>():
    logger().info("U16ValueVector found!");
    diffPoints = getDifference<U32ValueVector>(first->mPoints, second->mPoints);
    break;
  case cs::utils::variantIndex<PointsType, I16ValueVector>():
    logger().info("I16ValueVector found!");
    diffPoints = getDifference<I16ValueVector>(first->mPoints, second->mPoints);
    break;
  case cs::utils::variantIndex<PointsType, I32ValueVector>():
    logger().info("I32ValueVector found!");
    diffPoints = getDifference<I32ValueVector>(first->mPoints, second->mPoints);
    break;
  case cs::utils::variantIndex<PointsType, F32ValueVector>():
    logger().info("F32ValueVector found!");
    diffPoints = getDifference<F32ValueVector>(first->mPoints, second->mPoints);
    break;
  default:
    logger().warn("Unexpected Type!");
    break;
  }

  // Fill output value
  mValue = std::make_shared<Image2D>(diffPoints,
      // Reuse first socket's metadata
      first->mNumScalars, first->mDimension, first->mBounds,
      // Reuse timestamp if both are identical, else drop timestamp
      first->mTimeStamp == second->mTimeStamp ? first->mTimeStamp : std::nullopt);

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