////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WCSCoverageImage.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../../csl-ogc/src/wcs/WebCoverageService.hpp"
#include "../../../csl-ogc/src/wcs/WebCoverageTextureLoader.hpp"
#include "../types/CoverageContainer.hpp"
#include "../types/types.hpp"
#include "../logger.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string WCSCoverageImage::sName = "WCSCoverageImage";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WCSCoverageImage::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/WCSCoverageImage.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<WCSCoverageImage> WCSCoverageImage::sCreate() {
  return std::make_unique<WCSCoverageImage>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WCSCoverageImage::WCSCoverageImage() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WCSCoverageImage::~WCSCoverageImage() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WCSCoverageImage::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSCoverageImage::onMessageFromJS(nlohmann::json const& message) {

  logger().debug("WCSCoverageImage: Message form JS: {}", message.dump());

  // process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json WCSCoverageImage::getData() const {
  return nlohmann::json();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSCoverageImage::setData(nlohmann::json const& json) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSCoverageImage::process() {
  auto coverage = readInput<std::shared_ptr<CoverageContainer>>("coverageIn", nullptr);
  if (coverage == nullptr) {
    return;
  }

  // create request for texture loading
  csl::ogc::WebCoverageTextureLoader::Request request = getRequest();

  // load texture
  auto texLoader  = csl::ogc::WebCoverageTextureLoader();
  auto textureOpt = texLoader.loadTexture(*coverage->mServer, *coverage->mImageChannel, request,
      "../../../install/windows-Release/share/cache/csp-visual-query/texture-cache", true);

  if (textureOpt.has_value()) {
    auto texture     = textureOpt.value();
    auto textureSize = texture.x * texture.y;

    Image2D image;
    image.mDimension = {texture.x, texture.y};
    image.mBounds    = {texture.lnglatBounds[0], texture.lnglatBounds[1], texture.lnglatBounds[2],
           texture.lnglatBounds[3]};

    switch (texture.type) {
    case 1: // UInt8
    {
      std::vector<uint8_t> textureData(static_cast<uint8_t*>(texture.buffer),
          static_cast<uint8_t*>(texture.buffer) + textureSize);

      U8ValueVector pointData{};

      for (uint8_t scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    case 2: // UInt16
    {
      std::vector<uint16_t> textureData(static_cast<uint16_t*>(texture.buffer),
          static_cast<uint16_t*>(texture.buffer) + textureSize);

      U16ValueVector pointData{};

      for (uint16_t scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    case 3: // Int16
    {
      std::vector<int16_t> textureData(static_cast<int16_t*>(texture.buffer),
          static_cast<int16_t*>(texture.buffer) + textureSize);

      I16ValueVector pointData{};

      for (int16_t scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    case 4: // UInt32
    {
      std::vector<uint32_t> textureData(static_cast<uint32_t*>(texture.buffer),
          static_cast<uint32_t*>(texture.buffer) + textureSize);

      U32ValueVector pointData{};

      for (uint32_t scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    case 5: // Int32
    {
      std::vector<int32_t> textureData(static_cast<int32_t*>(texture.buffer),
          static_cast<int32_t*>(texture.buffer) + textureSize);

      I32ValueVector pointData{};

      for (int32_t scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    case 6: // Float32
    case 7: {
      std::vector<float> textureData(static_cast<float*>(texture.buffer),
          static_cast<float*>(texture.buffer) + textureSize);

      F32ValueVector pointData{};

      for (float scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    default:
      logger().error("Texture has no known data type.");
    }

    writeOutput("imageOut", std::make_shared<Image2D>(image));
  }
}

csl::ogc::WebCoverageTextureLoader::Request WCSCoverageImage::getRequest() {
  csl::ogc::WebCoverageTextureLoader::Request request;

  // request.mTime = std::to_string(readInput<double>("wcsTimeIn", 0.0));

  csl::ogc::Bounds bound;
  bound.mMinLon   = readInput<double>("lngBoundMinIn", -180.);
  bound.mMaxLon   = readInput<double>("lngBoundMaxIn", 180.);
  bound.mMinLat   = readInput<double>("latBoundMinIn", -90.);
  bound.mMaxLat   = readInput<double>("latBoundMaxIn", 90.);
  request.mBounds = bound;

  request.mMaxSize = readInput<int>("resolutionIn", 1024);

  request.mFormat = "image/tiff";

  return request;
}

} // namespace csp::visualquery