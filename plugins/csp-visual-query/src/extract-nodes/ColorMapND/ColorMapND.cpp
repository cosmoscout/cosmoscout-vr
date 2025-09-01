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
  // mCurrentOperation = message.at("text").get<std::string>();
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

  logger().info("Texture loaded: {}x{}x{}", texture.mWidth, texture.mHeight, texture.mBands);

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

  // Print the band ranges
  for (size_t i = 0; i < bandRanges.size(); i++) {
    logger().info("Band {}: {} - {}", i, bandRanges[i].x, bandRanges[i].y);
  }

  forEachValue([&](uint32_t i, float& value) {
    // Ignore no data values.
    if (value > 0.0) {
      value = (value - bandRanges[i].x) / (bandRanges[i].y - bandRanges[i].x);
    }
  });

  // We compute the 2D position of each data point in the color map space as the weighted average
  // of the band directions.
  std::vector<glm::vec2> dimensionDirections(texture.mBands);
  for (uint32_t i = 0; i < texture.mBands; i++) {
    float alpha            = (i / static_cast<float>(texture.mBands)) * 2.f * glm::pi<float>();
    dimensionDirections[i] = glm::vec2(std::cos(alpha), std::sin(alpha));
  }

  std::vector<glm::vec2> pointPositions(texture.mWidth * texture.mHeight);
  forEachPixel([&](uint32_t x, uint32_t y, std::vector<float> const& values) {
    glm::vec2 position(0.0, 0.0);
    float     sum = 0.0;
    for (uint32_t i = 0; i < texture.mBands; i++) {
      sum += values[i];
      position += values[i] * dimensionDirections[i];
    }
    size_t index = y * texture.mWidth + x;

    if (sum > 0.0) {
      position /= sum;
    }

    pointPositions[index] = position;
  });

  for (size_t i = 0; i < pointPositions.size(); i += 1000) {
    if (pointPositions[i].x != 0.0 || pointPositions[i].y != 0) {
      logger().info("{} {}", pointPositions[i].x, pointPositions[i].y);
    }
  }

  // For now, we just calculate the magnitude of the vector and write this as a grayscale image.
  F32ValueVector pointData(texture.mWidth * texture.mHeight);

  forEachPixel([&](uint32_t x, uint32_t y, std::vector<float> const& values) {
    float magnitude = 0.0;
    for (uint32_t i = 0; i < texture.mBands; i++) {
      magnitude += values[i] * values[i];
    }
    magnitude    = std::sqrt(magnitude);
    size_t index = y * texture.mWidth + x;

    pointData[index].push_back(magnitude);
    pointData[index].push_back(magnitude);
    pointData[index].push_back(magnitude);
  });

  image.mPoints = pointData;

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