////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WCSImageRGBA.hpp"

#include "../../../../csl-ogc/src/wcs/WebCoverageService.hpp"
#include "../../../../csl-ogc/src/wcs/WebCoverageTextureLoader.hpp"
#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../logger.hpp"
#include "../../types/CoverageContainer.hpp"
#include "../../types/types.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
template <typename T>
std::vector<std::vector<T>> convertImageData(csl::ogc::GDALReader::Texture const& texture) {
  std::vector<std::vector<T>> pointData(texture.mWidth * texture.mHeight);
  uint32_t                    bandStride = texture.mWidth * texture.mHeight;

  logger().info("Texture has {} bands.", texture.mBands);

  for (uint32_t y = 0; y < texture.mHeight; y++) {
    for (uint32_t x = 0; x < texture.mWidth; x++) {
      uint32_t index = (y * texture.mWidth + x);
      for (uint32_t band = 0; band < texture.mBands; band++) {
        pointData[index].push_back(*(static_cast<T*>(texture.mBuffer) + band * bandStride + index));
      }
    }
  }

  return pointData;
}
} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string WCSImageRGBA::sName = "WCSImageRGBA";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WCSImageRGBA::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/WCSImageRGBA.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<WCSImageRGBA> WCSImageRGBA::sCreate() {
  return std::make_unique<WCSImageRGBA>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WCSImageRGBA::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSImageRGBA::process() {
  auto coverage = readInput<std::shared_ptr<CoverageContainer>>("coverageIn", nullptr);
  if (coverage == nullptr) {
    return;
  }

  logger().info("Processing WCSImageRGBA node.");

  // create request for texture loading
  csl::ogc::WebCoverageTextureLoader::Request request = getRequest();

  // load texture
  auto texLoader  = csl::ogc::WebCoverageTextureLoader();
  auto textureOpt = texLoader.loadTexture(
      *coverage->mServer, *coverage->mImageChannel, request, "wcs-cache", true);

  if (textureOpt.has_value()) {
    auto texture = textureOpt.value();

    Image2D image;
    image.mNumScalars = texture.mBands;
    image.mDimension  = {texture.mWidth, texture.mHeight};

    // convert radians to degree
    image.mBounds = {texture.mLnglatBounds[0] * (180 / M_PI),
        texture.mLnglatBounds[2] * (180 / M_PI), texture.mLnglatBounds[3] * (180 / M_PI),
        texture.mLnglatBounds[1] * (180 / M_PI)};

    switch (texture.mDataType) {
    case 1: // UInt8
      image.mPoints.emplace<U8ValueVector>(convertImageData<uint8_t>(texture));
      break;

    case 2: // UInt16
      image.mPoints.emplace<U16ValueVector>(convertImageData<uint16_t>(texture));
      break;

    case 3: // Int16
      image.mPoints.emplace<I16ValueVector>(convertImageData<int16_t>(texture));
      break;

    case 4: // UInt32
      image.mPoints.emplace<U32ValueVector>(convertImageData<uint32_t>(texture));
      break;

    case 5: // Int32
      image.mPoints.emplace<I32ValueVector>(convertImageData<int32_t>(texture));
      break;

    case 6: // Float32
    case 7:
      image.mPoints.emplace<F32ValueVector>(convertImageData<float>(texture));
      break;

    default:
      logger().error("Texture has no known data type.");
    }

    writeOutput("imageOut", std::make_shared<Image2D>(image));
  }
}

csl::ogc::WebCoverageTextureLoader::Request WCSImageRGBA::getRequest() {
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

  request.mMaxSize  = readInput<int>("resolutionIn", 1024);
  request.mBandList = {1, 2, 3, 4};

  request.mFormat = "image/tiff";

  return request;
}

} // namespace csp::visualquery