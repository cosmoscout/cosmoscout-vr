////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "OverlayRender.hpp"

#include "../../logger.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../../../src/cs-utils/utils.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

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
    std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::Settings>    settings) {
  return std::make_unique<OverlayRender>(std::move(solarSystem), std::move(settings));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& OverlayRender::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OverlayRender::OverlayRender(std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::Settings>                             settings)
    : mSolarSystem(std::move(solarSystem))
    , mSettings(std::move(settings)) {
  mRenderer = std::make_unique<Renderer>(mSolarSystem, mSettings);

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), mRenderer.get()));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::ePlanets) + 10);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OverlayRender::~OverlayRender() = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json OverlayRender::getData() const {
  nlohmann::json data;

  std::set<std::string> objects{};

  for (const auto& item : mSettings->mObjects) {
    objects.insert(item.first);
  }

  std::vector<std::string> list{objects.begin(), objects.end()};
  list.insert(list.begin(), "None");

  data["options"]      = list;
  data["selectedBody"] = mRenderer->getObject();

  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OverlayRender::setData(nlohmann::json const& json) {
  if (json.find("selectedBody") != json.end()) {
    mRenderer->setObject(json["selectedBody"]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OverlayRender::init() {
  std::set<std::string> centerNames{};

  for (const auto& item : mSettings->mObjects) {
    centerNames.insert(item.second->getCenterName());
  }

  std::vector<std::string> data{centerNames.begin(), centerNames.end()};
  data.insert(data.begin(), "None");

  sendMessageToJS(data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OverlayRender::onMessageFromJS(const nlohmann::json& message) {
  mRenderer->setObject(message.at("text").get<std::string>());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Image2D normalize(Image2D const& image, glm::dvec2 minMax) {
  if (std::holds_alternative<F32ValueVector>(image.mPoints)) {
    Image2D copy{image};
    auto& points = std::get<F32ValueVector>(copy.mPoints);
    for (auto && point : points) {
      for (auto && scalar : point) {
        scalar = static_cast<float>((static_cast<double>(scalar) - minMax.x) / (minMax.y - minMax.x));
      }
    }

    return copy;
  }
  return image;
}

void OverlayRender::process() {
  auto input = readInput<std::shared_ptr<Image2D>>("Image2D", nullptr);
  if (input == nullptr) {
    return;
  }

  auto minMax = readInput<std::pair<double, double>>("minMax", std::pair(input->mMinMax.x, input->mMinMax.y));

  mRenderer->setData(std::make_shared<Image2D>(normalize(*input, glm::dvec2(minMax.first, minMax.second))));
  mRenderer->setMinMax(glm::vec2(0.F, 1.F));

  auto lut = readInput<std::vector<glm::vec4>>("lut", {});
  // use a grayscale transfer function if none is connected
  if (lut.empty()) {
    for (int i = 0; i < 256; i++) {
      float value = static_cast<float>(i) / 255.0f;
      lut.push_back(glm::vec4(value, value, value, 1.f));
    }
  }
  mRenderer->setLUT(lut);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery