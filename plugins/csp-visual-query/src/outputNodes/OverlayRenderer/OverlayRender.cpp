////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "OverlayRender.hpp"

#include "../../../../../src/cs-utils/filesystem.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string OverlayRender::sName = "OverlayRender";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string OverlayRender::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/OverlayRender.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<OverlayRender> OverlayRender::sCreate(
    std::shared_ptr<cs::core::SolarSystem> solarSystem) {
  return std::make_unique<OverlayRender>(std::move(solarSystem));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& OverlayRender::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OverlayRender::OverlayRender(std::shared_ptr<cs::core::SolarSystem> solarSystem)
    : mSolarSystem(std::move(solarSystem)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OverlayRender::~OverlayRender() = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

void OverlayRender::process() {
  auto input = readInput<std::shared_ptr<Image2D>>("Image2D", nullptr);
  if (!input) {
    return;
  }

  if (std::holds_alternative<U8ValueVector>(input->mPoints)) {
    for (auto const& entry : std::get<U8ValueVector>(input->mPoints)) {
      logger().info(entry.at(0));
    }
  } else if (std::holds_alternative<U16ValueVector>(input->mPoints)) {
    for (auto const& entry : std::get<U16ValueVector>(input->mPoints)) {
      logger().info(entry.at(0));
    }
  } else if (std::holds_alternative<U32ValueVector>(input->mPoints)) {
    for (auto const& entry : std::get<U32ValueVector>(input->mPoints)) {
      logger().info(entry.at(0));
    }
  } else if (std::holds_alternative<I16ValueVector>(input->mPoints)) {
    for (auto const& entry : std::get<I16ValueVector>(input->mPoints)) {
      logger().info(entry.at(0));
    }
  } else if (std::holds_alternative<I32ValueVector>(input->mPoints)) {
    for (auto const& entry : std::get<I32ValueVector>(input->mPoints)) {
      logger().info(entry.at(0));
    }
  } else if (std::holds_alternative<F32ValueVector>(input->mPoints)) {
    for (auto const& entry : std::get<F32ValueVector>(input->mPoints)) {
      logger().info(entry.at(0));
    }
  } else {
    logger().error("Unknown type!");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
