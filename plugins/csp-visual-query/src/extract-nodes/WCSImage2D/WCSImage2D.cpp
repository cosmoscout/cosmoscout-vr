////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WCSImage2D.hpp"

#include "../../../../csl-ogc/src/wcs/WebCoverageService.hpp"
#include "../../../../csl-ogc/src/wcs/WebCoverageTextureLoader.hpp"
#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../logger.hpp"
#include "../../types/CoverageContainer.hpp"
#include "../../types/types.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string WCSImage2D::sName = "WCSImage2D";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WCSImage2D::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/WCSImage2D.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<WCSImage2D> WCSImage2D::sCreate() {
  return std::make_unique<WCSImage2D>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WCSImage2D::WCSImage2D() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WCSImage2D::~WCSImage2D() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WCSImage2D::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSImage2D::onMessageFromJS(nlohmann::json const& message) {

  logger().debug("WCSImage2D: Message form JS: {}", message.dump());

  // process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json WCSImage2D::getData() const {
  return nlohmann::json();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSImage2D::setData(nlohmann::json const& json) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSImage2D::process() {
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
    auto texture     = textureOpt.value();
    auto textureSize = texture.mWidth * texture.mHeight;

    Image2D image;
    image.mNumScalars = 1;
    image.mDimension  = {texture.mWidth, texture.mHeight};

    // convert radians to degree
    image.mBounds = {texture.mLnglatBounds[0] * (180 / M_PI),
        texture.mLnglatBounds[2] * (180 / M_PI), texture.mLnglatBounds[3] * (180 / M_PI),
        texture.mLnglatBounds[1] * (180 / M_PI)};

    switch (texture.mDataType) {
    case 1: // UInt8
    {
      std::vector<uint8_t> textureData(static_cast<uint8_t*>(texture.mBuffer),
          static_cast<uint8_t*>(texture.mBuffer) + textureSize);

      U8ValueVector pointData{};

      for (uint8_t scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    case 2: // UInt16
    {
      std::vector<uint16_t> textureData(static_cast<uint16_t*>(texture.mBuffer),
          static_cast<uint16_t*>(texture.mBuffer) + textureSize);

      U16ValueVector pointData{};

      for (uint16_t scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    case 3: // Int16
    {
      std::vector<int16_t> textureData(static_cast<int16_t*>(texture.mBuffer),
          static_cast<int16_t*>(texture.mBuffer) + textureSize);

      I16ValueVector pointData{};

      for (int16_t scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    case 4: // UInt32
    {
      std::vector<uint32_t> textureData(static_cast<uint32_t*>(texture.mBuffer),
          static_cast<uint32_t*>(texture.mBuffer) + textureSize);

      U32ValueVector pointData{};

      for (uint32_t scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    case 5: // Int32
    {
      std::vector<int32_t> textureData(static_cast<int32_t*>(texture.mBuffer),
          static_cast<int32_t*>(texture.mBuffer) + textureSize);

      I32ValueVector pointData{};

      for (int32_t scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    case 6: // Float32
    case 7: {
      std::vector<float> textureData(
          static_cast<float*>(texture.mBuffer), static_cast<float*>(texture.mBuffer) + textureSize);

      F32ValueVector pointData{};

      for (float& scalar : textureData) {
        // scalar *= 1'000'000'000;
        pointData.emplace_back(std::vector{scalar});
      }

      auto result = std::minmax_element(textureData.begin(), textureData.end());
      logger().info("WCSImage2D: Min: {}, Max: {}", *(result.first), *(result.second));
      image.mMinMax = glm::dvec2(*(result.first), *(result.second));

      image.mPoints = pointData;
      break;
    }

    default:
      logger().error("Texture has no known data type.");
    }

    writeOutput("imageOut", std::make_shared<Image2D>(image));
  }
}

csl::ogc::WebCoverageTextureLoader::Request WCSImage2D::getRequest() {
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

  request.mMaxSize    = readInput<int>("resolutionIn", 1024);
  request.mLayerRange = std::make_pair(readInput<int>("layerIn", 1), readInput<int>("layerIn", 1));

  request.mFormat = "image/tiff";

  return request;
}

} // namespace csp::visualquery