////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ColorMapND.hpp"

#include "../../../../csl-ogc/src/wcs/WebCoverageService.hpp"
#include "../../../../csl-ogc/src/wcs/WebCoverageTextureLoader.hpp"
#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../logger.hpp"
#include "../../types/CoverageContainer.hpp"
#include "../../types/types.hpp"

namespace csp::visualquery {

// From https://github.com/gka/chroma.js. Licensed under BSD by Gregor Aisch.
namespace {

float compand(float linear) {
  float sign = linear < 0 ? -1.0f : 1.0f;
  linear     = std::abs(linear);
  return (
      (linear <= 0.0031308 ? linear * 12.92 : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055) * sign);
}

glm::vec3 xyz2rgb(float x, float y, float z) {
  const float MtxAdaptMa_m00 = 0.8951;
  const float MtxAdaptMa_m01 = -0.7502;
  const float MtxAdaptMa_m02 = 0.0389;
  const float MtxAdaptMa_m10 = 0.2664;
  const float MtxAdaptMa_m11 = 1.7135;
  const float MtxAdaptMa_m12 = -0.0685;
  const float MtxAdaptMa_m20 = -0.1614;
  const float MtxAdaptMa_m21 = 0.0367;
  const float MtxAdaptMa_m22 = 1.0296;

  const float MtxAdaptMaI_m00 = 0.9869929054667123;
  const float MtxAdaptMaI_m01 = 0.43230526972339456;
  const float MtxAdaptMaI_m02 = -0.008528664575177328;
  const float MtxAdaptMaI_m10 = -0.14705425642099013;
  const float MtxAdaptMaI_m11 = 0.5183602715367776;
  const float MtxAdaptMaI_m12 = 0.04004282165408487;
  const float MtxAdaptMaI_m20 = 0.15996265166373125;
  const float MtxAdaptMaI_m21 = 0.0492912282128556;
  const float MtxAdaptMaI_m22 = 0.9684866957875502;

  const float MtxXYZ2RGB_m00 = 3.2404541621141045;
  const float MtxXYZ2RGB_m01 = -0.9692660305051868;
  const float MtxXYZ2RGB_m02 = 0.055643430959114726;
  const float MtxXYZ2RGB_m10 = -1.5371385127977166;
  const float MtxXYZ2RGB_m11 = 1.8760108454466942;
  const float MtxXYZ2RGB_m12 = -0.2040259135167538;
  const float MtxXYZ2RGB_m20 = -0.498531409556016;
  const float MtxXYZ2RGB_m21 = 0.041556017530349834;
  const float MtxXYZ2RGB_m22 = 1.0572251882231791;

  const float RefWhiteRGB_X = 0.95047;
  const float RefWhiteRGB_Y = 1;
  const float RefWhiteRGB_Z = 1.08883;

  const float Xn = 0.95047;
  const float Yn = 1;
  const float Zn = 1.08883;

  float As = Xn * MtxAdaptMa_m00 + Yn * MtxAdaptMa_m10 + Zn * MtxAdaptMa_m20;
  float Bs = Xn * MtxAdaptMa_m01 + Yn * MtxAdaptMa_m11 + Zn * MtxAdaptMa_m21;
  float Cs = Xn * MtxAdaptMa_m02 + Yn * MtxAdaptMa_m12 + Zn * MtxAdaptMa_m22;

  float Ad = RefWhiteRGB_X * MtxAdaptMa_m00 + RefWhiteRGB_Y * MtxAdaptMa_m10 +
             RefWhiteRGB_Z * MtxAdaptMa_m20;
  float Bd = RefWhiteRGB_X * MtxAdaptMa_m01 + RefWhiteRGB_Y * MtxAdaptMa_m11 +
             RefWhiteRGB_Z * MtxAdaptMa_m21;
  float Cd = RefWhiteRGB_X * MtxAdaptMa_m02 + RefWhiteRGB_Y * MtxAdaptMa_m12 +
             RefWhiteRGB_Z * MtxAdaptMa_m22;

  float X1 = (x * MtxAdaptMa_m00 + y * MtxAdaptMa_m10 + z * MtxAdaptMa_m20) * (Ad / As);
  float Y1 = (x * MtxAdaptMa_m01 + y * MtxAdaptMa_m11 + z * MtxAdaptMa_m21) * (Bd / Bs);
  float Z1 = (x * MtxAdaptMa_m02 + y * MtxAdaptMa_m12 + z * MtxAdaptMa_m22) * (Cd / Cs);

  float X2 = X1 * MtxAdaptMaI_m00 + Y1 * MtxAdaptMaI_m10 + Z1 * MtxAdaptMaI_m20;
  float Y2 = X1 * MtxAdaptMaI_m01 + Y1 * MtxAdaptMaI_m11 + Z1 * MtxAdaptMaI_m21;
  float Z2 = X1 * MtxAdaptMaI_m02 + Y1 * MtxAdaptMaI_m12 + Z1 * MtxAdaptMaI_m22;

  float r = compand(X2 * MtxXYZ2RGB_m00 + Y2 * MtxXYZ2RGB_m10 + Z2 * MtxXYZ2RGB_m20);
  float g = compand(X2 * MtxXYZ2RGB_m01 + Y2 * MtxXYZ2RGB_m11 + Z2 * MtxXYZ2RGB_m21);
  float b = compand(X2 * MtxXYZ2RGB_m02 + Y2 * MtxXYZ2RGB_m12 + Z2 * MtxXYZ2RGB_m22);

  return {r, g, b};
}

glm::vec3 OKLab_to_XYZ(glm::vec3 const& OKLab) {
  // Given OKLab, convert to XYZ relative to D65
  glm::mat3 LMStoXYZ = {{1.2268798758459243, -0.5578149944602171, 0.2813910456659647},
      {-0.0405757452148008, 1.112286803280317, -0.0717110580655164},
      {-0.0763729366746601, -0.4214933324022432, 1.5869240198367816}};
  LMStoXYZ           = glm::transpose(LMStoXYZ);

  glm::mat3 OKLabtoLMS = {{1.0, 0.3963377773761749, 0.2158037573099136},
      {1.0, -0.1055613458156586, -0.0638541728258133},
      {1.0, -0.0894841775298119, -1.2914855480194092}};
  OKLabtoLMS           = glm::transpose(OKLabtoLMS);

  glm::vec3 LMSnl = OKLabtoLMS * OKLab;
  return LMStoXYZ * glm::pow(LMSnl, glm::vec3(3.f));
}

glm::vec3 oklab2rgb(float L, float a, float b) {
  auto XYZ = OKLab_to_XYZ({L, a, b});
  return xyz2rgb(XYZ[0], XYZ[1], XYZ[2]);
}

glm::vec3 lch2lab(float l, float c, float h) {
  h = glm::radians(h);
  return {l, std::cos(h) * c, std::sin(h) * c};
}

glm::vec3 oklch2rgb(float l, float c, float h) {
  auto Lab = lch2lab(l, c, h);
  return oklab2rgb(Lab[0], Lab[1], Lab[2]);
}
} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string ColorMapND::sName = "ColorMapND";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string ColorMapND::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/ColorMapND.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<ColorMapND> ColorMapND::sCreate() {
  return std::make_unique<ColorMapND>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& ColorMapND::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

void ColorMapND::init() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ColorMapND::onMessageFromJS(nlohmann::json const& message) {
  std::string operation = message.at("operation").get<std::string>();

  if (operation == "setDimensions") {
    this->mDimensionAngles = message.at("dimensions").get<std::vector<double>>();
  }

  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json ColorMapND::getData() const {
  nlohmann::json data;

  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ColorMapND::setData(nlohmann::json const& json) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ColorMapND::process() {
  auto coverage = readInput<std::shared_ptr<CoverageContainer>>("coverageIn", nullptr);
  if (coverage == nullptr) {
    return;
  }
  // create request for texture loading
  csl::ogc::WebCoverageTextureLoader::Request request = getRequest();

  // load texture
  auto texLoader  = csl::ogc::WebCoverageTextureLoader();
  auto textureOpt = texLoader.loadTexture(
      *coverage->mServer, *coverage->mImageChannel, request, "wcs-cache", true);

  if (!textureOpt.has_value()) {
    return;
  }

  auto texture = textureOpt.value();

  if (texture.mDataType != 6) {
    logger().error("Texture has wrong data type.");
    return;
  }

  // Lambda to apply a function to each data point in the texture. The band index and the value are
  // passed to the function.
  auto forEachValue = [&](std::function<void(uint32_t, float&)> func) {
    for (uint32_t y = 0; y < texture.mHeight; y++) {
      for (uint32_t x = 0; x < texture.mWidth; x++) {
        uint32_t index = (y * texture.mWidth + x);

        for (uint32_t i = 0; i < texture.mBands; i++) {
          float value = *(
              static_cast<float*>(texture.mBuffer) + i * texture.mWidth * texture.mHeight + index);
          func(i, value);
          *(static_cast<float*>(texture.mBuffer) + i * texture.mWidth * texture.mHeight + index) =
              value;
        }
      }
    }
  };

  // Lambda to apply a function to each pixel in the texture. The x and y coordinates as well as the
  // band values are passed to the function.
  auto forEachPixel = [&](std::function<void(uint32_t, uint32_t, std::vector<float> const&)> func) {
    for (uint32_t y = 0; y < texture.mHeight; y++) {
      for (uint32_t x = 0; x < texture.mWidth; x++) {
        uint32_t index = (y * texture.mWidth + x);

        std::vector<float> values(texture.mBands);
        for (uint32_t i = 0; i < texture.mBands; i++) {
          values[i] = *(
              static_cast<float*>(texture.mBuffer) + i * texture.mWidth * texture.mHeight + index);
        }

        func(x, y, values);
      }
    }
  };

  Image2D image;
  image.mDimension  = {texture.mWidth, texture.mHeight};
  image.mMinMax     = {0.0, 1.0};
  image.mNumScalars = 3;
  image.mBounds = {texture.mLnglatBounds[0] * (180 / M_PI), texture.mLnglatBounds[2] * (180 / M_PI),
      texture.mLnglatBounds[3] * (180 / M_PI), texture.mLnglatBounds[1] * (180 / M_PI)};

  // First normalize the individual bands.
  std::vector<glm::vec2> bandRanges(texture.mBands,
      glm::vec2(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()));

  forEachValue([&](uint32_t i, float& value) {
    // Ignore no data values.
    if (value > 0.0) {
      bandRanges[i].x = std::min(bandRanges[i].x, value);
      bandRanges[i].y = std::max(bandRanges[i].y, value);
    }
  });

  forEachValue([&](uint32_t i, float& value) {
    // Ignore no data values.
    if (value > 0.0) {
      value = (value - bandRanges[i].x) / (bandRanges[i].y - bandRanges[i].x);
    }
  });

  // We compute the 2D position of each data point in the color map space as the weighted average
  // of the band directions.
  if (this->mDimensionAngles.size() != texture.mBands) {
    this->mDimensionAngles.resize(texture.mBands);

    for (uint32_t i = 0; i < texture.mBands; i++) {
      this->mDimensionAngles[i] = (i / static_cast<float>(texture.mBands)) * 2.f * glm::pi<float>();
    }
  }

  std::vector<glm::vec2> dimensionDirections(texture.mBands);
  for (uint32_t i = 0; i < texture.mBands; i++) {
    dimensionDirections[i] =
        glm::vec2(std::cos(this->mDimensionAngles[i]), std::sin(this->mDimensionAngles[i]));
  }

  F32ValueVector                  pointColors(texture.mWidth * texture.mHeight);
  std::vector<std::vector<float>> samples;

  forEachPixel([&](uint32_t x, uint32_t y, std::vector<float> const& values) {
    glm::vec2 position(0.0, 0.0);
    float     sum = 0.0;
    for (uint32_t i = 0; i < texture.mBands; i++) {
      float weight = std::pow(values[i], 2.0f);
      sum += weight;
      position += weight * dimensionDirections[i];
    }

    size_t index = y * texture.mWidth + x;

    if (sum > 0.0) {
      position /= sum;
    }

    float     L         = 0.7;
    float     maxChroma = 0.3;
    float     c         = maxChroma * glm::length(position);
    float     h         = glm::degrees(std::atan2(position.y, position.x)) + 360.0f;
    glm::vec3 rgb       = oklch2rgb(L, c, h);

    pointColors[index] = {rgb.r, rgb.g, rgb.b};

    if (index % 1000 == 0) {
      samples.emplace_back(values);
    }
  });

  nlohmann::json json;
  json["data"]["points"]     = samples;
  json["data"]["dimensions"] = this->mDimensionAngles;
  sendMessageToJS(json);

  image.mPoints = pointColors;
  writeOutput("imageOut", std::make_shared<Image2D>(image));
}

csl::ogc::WebCoverageTextureLoader::Request ColorMapND::getRequest() {
  csl::ogc::WebCoverageTextureLoader::Request request;

  request.mTime = readInput<std::string>("wcsTimeIn", "");
  if (request.mTime.value() == "") {
    request.mTime.reset();
  }

  auto bounds = readInput<std::array<double, 4>>("boundsIn", {-180., 180., -90., 90.});

  csl::ogc::Bounds2D bound;
  request.mBounds.mMinLon = bounds[0];
  request.mBounds.mMaxLon = bounds[1];
  request.mBounds.mMinLat = bounds[2];
  request.mBounds.mMaxLat = bounds[3];

  request.mMaxSize = readInput<int>("resolutionIn", 1024);
  request.mFormat  = "image/tiff";

  return request;
}

} // namespace csp::visualquery