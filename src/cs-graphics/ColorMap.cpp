////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ColorMap.hpp"

#include "../cs-utils/doctest.hpp"

#include <nlohmann/json.hpp>

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

std::vector<glm::vec4> mergeColorMapData(ColorMapData colorMapData, int resolution) {
  std::vector<glm::vec4> colors(resolution);
  for (size_t i(0); i < colors.size(); ++i) {
    float     key   = static_cast<float>(i) / static_cast<float>(colors.size() - 1);
    glm::vec3 color = interpolate(key, colorMapData.rgbStops);
    float     alpha = interpolate(key, colorMapData.alphaStops);
    colors[i]       = glm::vec4(color, alpha);
  }

  return colors;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<VistaTexture> generateTexture(std::vector<glm::vec4> colors) {
  std::unique_ptr<VistaTexture> texture = std::make_unique<VistaTexture>(GL_TEXTURE_1D);

  texture->UploadTexture((int)colors.size(), 1, colors.data(), false, GL_RGBA, GL_FLOAT);

  texture->SetWrapS(GL_CLAMP_TO_EDGE);
  texture->SetWrapT(GL_CLAMP_TO_EDGE);

  texture->SetMinFilter(GL_LINEAR);
  texture->SetMagFilter(GL_LINEAR);

  return texture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

ColorMap::ColorMap(std::string const& sJsonString) {
  nlohmann::json json         = nlohmann::json::parse(sJsonString);
  ColorMapData   colorMapData = json;
  mRawData                    = mergeColorMapData(colorMapData, mResolution);
  mTexture                    = generateTexture(mRawData);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ColorMap::ColorMap(boost::filesystem::path const& sJsonPath) {
  std::ifstream  file(sJsonPath.string());
  nlohmann::json json;
  file >> json;
  ColorMapData colorMapData = json;
  mRawData                  = mergeColorMapData(colorMapData, mResolution);
  mTexture                  = generateTexture(mRawData);
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

std::vector<glm::vec4> ColorMap::getRawData() {
  return mRawData;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("cs::graphics::ColorMap::interpolate") {
  std::map<float, glm::vec3> stops = {{0.F, glm::vec3(1.F, 1.F, 0.F)},
      {0.5F, glm::vec3(1.F, 0.F, 0.F)}, {1.0F, glm::vec3(1.F, 0.F, 1.F)}};

  CHECK(interpolate(0.F, stops) == glm::vec3(1.F, 1.F, 0.F));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
