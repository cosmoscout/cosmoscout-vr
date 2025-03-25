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

  if (textureOpt.has_value()) {
    auto texture = textureOpt.value();

    logger().info("Texture loaded: {}x{}x{}", texture.mWidth, texture.mHeight, texture.mBands);

    Image2D image;
    image.mDimension  = {texture.mWidth, texture.mHeight};
    image.mMinMax     = {0.0, 1.0};
    image.mNumScalars = 3;

    // convert radians to degree
    image.mBounds = {texture.mLnglatBounds[0] * (180 / M_PI),
        texture.mLnglatBounds[2] * (180 / M_PI), texture.mLnglatBounds[3] * (180 / M_PI),
        texture.mLnglatBounds[1] * (180 / M_PI)};

    if (texture.mDataType == 6) {

      F32ValueVector pointData(texture.mWidth * texture.mHeight);
      uint32_t       bandStride = texture.mWidth * texture.mHeight;
      float          maxLength  = 0.0;

      for (uint32_t y = 0; y < texture.mHeight; y++) {
        for (uint32_t x = 0; x < texture.mWidth; x++) {
          uint32_t index = (y * texture.mWidth + x);

          float length = 0.0;
          for (uint32_t i = 0; i < texture.mBands; i++) {
            float value = *(static_cast<float*>(texture.mBuffer) + i * bandStride + index);
            length += value * value;
          }
          length    = std::sqrt(length);
          maxLength = std::max(maxLength, length);

          pointData[index].push_back(length);
          pointData[index].push_back(0.0);
          pointData[index].push_back(0.0);
        }
      }

      for (uint32_t y = 0; y < texture.mHeight; y++) {
        for (uint32_t x = 0; x < texture.mWidth; x++) {
          uint32_t index = (y * texture.mWidth + x);
          pointData[index][0] /= maxLength;
        }
      }

      image.mPoints = pointData;

    } else {
      logger().error("Texture has wrong data type.");
    }

    writeOutput("imageOut", std::make_shared<Image2D>(image));
  }
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