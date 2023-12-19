////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "JsonVolumeFileLoader.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../logger.hpp"

#include <fstream>

namespace csp::visualquery {

/////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, JsonVolume& volume) {
  cs::core::Settings::deserialize(j, "dimensions", volume.dimensions);
  cs::core::Settings::deserialize(j, "origin", volume.origin);
  cs::core::Settings::deserialize(j, "spacing", volume.spacing);

  for (const auto& item : j.items()) {
    std::string key = item.key();
    if (key != "dimensions" && key != "origin" && key != "spacing" && key != "filename" &&
        item.value().is_array()) {
      cs::core::Settings::deserialize(j, key, volume.data);
      break;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string JsonVolumeFileLoader::sName = "JsonVolumeFileLoader";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string JsonVolumeFileLoader::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/JsonVolumeFileLoader.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<JsonVolumeFileLoader> JsonVolumeFileLoader::sCreate() {
  return std::make_unique<JsonVolumeFileLoader>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

JsonVolumeFileLoader::JsonVolumeFileLoader() noexcept {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

JsonVolumeFileLoader::~JsonVolumeFileLoader() noexcept = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& JsonVolumeFileLoader::getName() const noexcept {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void JsonVolumeFileLoader::process() noexcept {
  writeOutput("Volume3D", mData);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec3 hueToRGB(float hue) {
  hue *= 360.F;
  float hPrime = std::fmod(hue / 60.F, 6.F);
  float x      = 1.F - std::fabs(std::fmod(hPrime, 2.F) - 1.F);

  float r;
  float g;
  float b;

  if (0 <= hPrime && hPrime < 1) {
    r = 1;
    g = x;
    b = 0;
  } else if (1 <= hPrime && hPrime < 2) {
    r = x;
    g = 1;
    b = 0;
  } else if (2 <= hPrime && hPrime < 3) {
    r = 0;
    g = 1;
    b = x;
  } else if (3 <= hPrime && hPrime < 4) {
    r = 0;
    g = x;
    b = 1;
  } else if (4 <= hPrime && hPrime < 5) {
    r = x;
    g = 0;
    b = 1;
  } else if (5 <= hPrime && hPrime < 6) {
    r = 1;
    g = 0;
    b = x;
  } else {
    r = 0;
    g = 0;
    b = 0;
  }

  return {r, g, b};
}

void JsonVolumeFileLoader::onMessageFromJS(nlohmann::json const& message) {
  cs::core::Settings::deserialize(message, "file", mFileName);

  std::ifstream f(mFileName);
  JsonVolume    v = nlohmann::json::parse(f);

  glm::dvec3         maxBounds = v.origin + (glm::dvec3(v.dimensions) * v.spacing);
  csl::ogc::Bounds3D bounds{
      v.origin.x, maxBounds.x, v.origin.y, maxBounds.y, v.origin.z, maxBounds.z};

  // For now we tranform this data to RGB right away, this can be skipped later
  auto   minMax = std::minmax_element(v.data.begin(), v.data.end());
  double min    = *minMax.first;
  double max    = *minMax.second;
  double range  = max - min;

  std::vector<std::vector<float>> points;
  for (double p : v.data) {
    points.emplace_back(std::vector{static_cast<float>((p - min) / range)});
  }

  mData = std::make_shared<Volume3D>(points, 1, v.dimensions, bounds);

  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json JsonVolumeFileLoader::getData() const {
  return {"file", mFileName};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void JsonVolumeFileLoader::setData(nlohmann::json const& json) {
  cs::core::Settings::deserialize(json, "file", mFileName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery