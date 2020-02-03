////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ColorMap.hpp"

#include "../cs-utils/doctest.hpp"

#include <glm/glm.hpp>
#include <json.hpp>

#include <fstream>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ColorMapData {
  std::map<float, glm::vec3> rgbStops;
  std::map<float, float>     alphaStops;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, ColorMapData& s) {
  auto rgb = j.at("RGB");
  for (nlohmann::json::iterator it = rgb.begin(); it != rgb.end(); ++it) {
    glm::vec3 value;
    for (int i = 0; i < 3; ++i) {
      value[i] = it.value().at(i);
    }

    s.rgbStops[std::stof(it.key())] = value;
  }

  auto alpha = j.at("Alpha");
  for (nlohmann::json::iterator it = alpha.begin(); it != alpha.end(); ++it) {
    s.alphaStops[std::stof(it.key())] = it.value();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T interpolate(float key, const std::map<float, T>& map) {
  if (map.find(key) != map.end()) {
    return map.at(key);
  }

  if (key < map.begin()->first) {
    return map.begin()->second;
  }

  if (key > map.rbegin()->first) {
    return map.rbegin()->second;
  }

  auto lower = map.lower_bound(key) == map.begin() ? map.begin() : --(map.lower_bound(key));
  auto upper = map.upper_bound(key);

  return lower->second + (upper->second - lower->second) * float(key - lower->first) /
                             std::fabs(upper->first - lower->first);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

ColorMap::ColorMap(std::string const& sJsonFile)
    : mTexture(new VistaTexture(GL_TEXTURE_1D)) {
  ColorMapData colorMapData;
  {
    std::ifstream  file(sJsonFile);
    nlohmann::json json;
    file >> json;
    colorMapData = json;
  }

  const int RESOLUTION = 256;

  std::vector<glm::vec4> colors(RESOLUTION);
  for (int i(0); i < colors.size(); ++i) {
    float     key   = static_cast<float>(i) / (colors.size() - 1);
    glm::vec3 color = interpolate(key, colorMapData.rgbStops);
    float     alpha = interpolate(key, colorMapData.alphaStops);
    colors[i]       = glm::vec4(color, alpha);
  }

  mTexture->UploadTexture(RESOLUTION, 1, colors.data(), false, GL_RGBA, GL_FLOAT);

  mTexture->SetWrapS(GL_CLAMP_TO_EDGE);
  mTexture->SetWrapT(GL_CLAMP_TO_EDGE);

  mTexture->SetMinFilter(GL_LINEAR);
  mTexture->SetMagFilter(GL_LINEAR);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ColorMap::bind(unsigned unit) {
  mTexture->Bind(unit);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ColorMap::unbind(unsigned unit) {
  mTexture->Unbind(unit);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("cs::graphics::ColorMap::interpolate") {
  std::map<float, glm::vec3> stops = {{0.f, glm::vec3(1.f, 1.f, 0.f)},
      {0.5f, glm::vec3(1.f, 0.f, 0.f)}, {1.0f, glm::vec3(1.f, 0.f, 1.f)}};

  CHECK(interpolate(0.f, stops) == glm::vec3(1.f, 1.f, 0.f));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
