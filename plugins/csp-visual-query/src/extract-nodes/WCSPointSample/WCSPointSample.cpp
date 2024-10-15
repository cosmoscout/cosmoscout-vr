////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WCSPointSample.hpp"

#include "../../../../csl-ogc/src/wcs/WebCoverageService.hpp"
#include "../../../../csl-ogc/src/wcs/WebCoverageTextureLoader.hpp"
#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../logger.hpp"
#include "../../types/CoverageContainer.hpp"
#include "../../types/types.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string WCSPointSample::sName = "WCSPointSample";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WCSPointSample::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/WCSPointSample.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<WCSPointSample> WCSPointSample::sCreate() {
  return std::make_unique<WCSPointSample>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WCSPointSample::WCSPointSample() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WCSPointSample::~WCSPointSample() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WCSPointSample::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSPointSample::onMessageFromJS(nlohmann::json const& message) {

  logger().debug("WCSPointSample: Message form JS: {}", message.dump());

  // process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json WCSPointSample::getData() const {
  return nlohmann::json();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSPointSample::setData(nlohmann::json const& json) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSPointSample::process() {
  auto coverage = readInput<std::shared_ptr<CoverageContainer>>("coverageIn", nullptr);
  if (coverage == nullptr) {
    return;
  }

  // create request for texture loading
  csl::ogc::WebCoverageTextureLoader::Request request = getRequest();

  // Load the texture. We do not add it to the cache, because it is very unlikely that we will
  // request the same exact location again.
  auto texLoader  = csl::ogc::WebCoverageTextureLoader();
  auto textureOpt = texLoader.loadTexture(
      *coverage->mServer, *coverage->mImageChannel, request, "wcs-cache", false);

  if (textureOpt.has_value()) {
    auto texture = textureOpt.value();

    std::cout << "Texture loaded: " << texture.mWidth << "x" << texture.mHeight << "x"
              << texture.mBands << std::endl;

    Image1D image;
    image.mNumScalars = 1;
    image.mDimension  = texture.mBands;

    auto coords = readInput<std::pair<double, double>>("coords", {0., 0.});

    // convert radians to degree
    image.mLongLat = {coords.first, coords.second};

    switch (texture.mDataType) {
    case 1: // UInt8
    {
      std::vector<uint8_t> textureData(static_cast<uint8_t*>(texture.mBuffer),
          static_cast<uint8_t*>(texture.mBuffer) + texture.mBands);

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
          static_cast<uint16_t*>(texture.mBuffer) + texture.mBands);

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
          static_cast<int16_t*>(texture.mBuffer) + texture.mBands);

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
          static_cast<uint32_t*>(texture.mBuffer) + texture.mBands);

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
          static_cast<int32_t*>(texture.mBuffer) + texture.mBands);

      I32ValueVector pointData{};

      for (int32_t scalar : textureData) {
        pointData.emplace_back(std::vector{scalar});
      }

      image.mPoints = pointData;
      break;
    }

    case 6: // Float32
    case 7: {
      std::vector<float> textureData(static_cast<float*>(texture.mBuffer),
          static_cast<float*>(texture.mBuffer) + texture.mBands);

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

    writeOutput("imageOut", std::make_shared<Image1D>(image));
  }
}

csl::ogc::WebCoverageTextureLoader::Request WCSPointSample::getRequest() {
  csl::ogc::WebCoverageTextureLoader::Request request;

  request.mTime = readInput<std::string>("wcsTimeIn", "");
  if (request.mTime.value() == "") {
    request.mTime.reset();
  }

  auto coords = readInput<std::pair<double, double>>("coords", {0., 0.});

  csl::ogc::Bounds2D bound;
  request.mBounds.mMinLon = coords.first;
  request.mBounds.mMaxLon = coords.first;
  request.mBounds.mMinLat = coords.second;
  request.mBounds.mMaxLat = coords.second;

  request.mMaxSize    = 1;
  request.mLayerRange = std::make_pair(1, 255);

  request.mFormat = "image/tiff";

  return request;
}

} // namespace csp::visualquery