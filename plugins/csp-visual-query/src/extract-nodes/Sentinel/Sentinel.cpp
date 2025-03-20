////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Sentinel.hpp"

#include "../../../../csl-ogc/src/wcs/WebCoverageService.hpp"
#include "../../../../csl-ogc/src/wcs/WebCoverageTextureLoader.hpp"
#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../logger.hpp"
#include "../../types/CoverageContainer.hpp"
#include "../../types/types.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string Sentinel::sName = "Sentinel";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Sentinel::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/Sentinel.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<Sentinel> Sentinel::sCreate() {
  return std::make_unique<Sentinel>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& Sentinel::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

void Sentinel::init() {

  std::vector<std::string> operations{
      "None",
      "True Color",
      "False Color Urban",
      "Moisture Index",
  };

  sendMessageToJS(operations);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Sentinel::onMessageFromJS(nlohmann::json const& message) {
  mCurrentOperation = message.at("text").get<std::string>();
  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json Sentinel::getData() const {
  nlohmann::json data;

  std::vector<std::string> operations{
      "None",
      "True Color",
      "False Color Urban",
      "Moisture Index",
  };

  data["operations"]        = operations;
  data["selectedOperation"] = mCurrentOperation;

  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Sentinel::setData(nlohmann::json const& json) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Sentinel::process() {
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

    Image2D image;
    image.mDimension = {texture.mWidth, texture.mHeight};

    if (mCurrentOperation == "Moisture Index") {
      image.mMinMax     = {-0.8, 0.8};
      image.mNumScalars = 1;
    } else if (mCurrentOperation == "False Color Urban") {
      image.mMinMax     = {0.0, 5000.0};
      image.mNumScalars = 3;
    } else {
      image.mMinMax     = {0.0, 3000.0};
      image.mNumScalars = 3;
    }

    // convert radians to degree
    image.mBounds = {texture.mLnglatBounds[0] * (180 / M_PI),
        texture.mLnglatBounds[2] * (180 / M_PI), texture.mLnglatBounds[3] * (180 / M_PI),
        texture.mLnglatBounds[1] * (180 / M_PI)};

    if (texture.mDataType == 6) {

      F32ValueVector pointData(texture.mWidth * texture.mHeight);
      uint32_t       bandStride = texture.mWidth * texture.mHeight;

      for (uint32_t y = 0; y < texture.mHeight; y++) {
        for (uint32_t x = 0; x < texture.mWidth; x++) {
          uint32_t index = (y * texture.mWidth + x);

          if (mCurrentOperation == "Moisture Index") {
            float b08 = *(static_cast<float*>(texture.mBuffer) + 0 * bandStride + index);
            float b11 = *(static_cast<float*>(texture.mBuffer) + 1 * bandStride + index);
            pointData[index].push_back((b08 - b11) / (b08 + b11));
          } else {
            float r = *(static_cast<float*>(texture.mBuffer) + 0 * bandStride + index);
            float g = *(static_cast<float*>(texture.mBuffer) + 1 * bandStride + index);
            float b = *(static_cast<float*>(texture.mBuffer) + 2 * bandStride + index);
            pointData[index].push_back(r);
            pointData[index].push_back(g);
            pointData[index].push_back(b);
          }
        }
      }

      image.mPoints = pointData;

    } else {
      logger().error("Texture has wrong data type.");
    }

    writeOutput("imageOut", std::make_shared<Image2D>(image));
  }
}

csl::ogc::WebCoverageTextureLoader::Request Sentinel::getRequest() {
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

  if (mCurrentOperation == "Moisture Index") {
    request.mBandList = {8, 11};
  } else if (mCurrentOperation == "False Color Urban") {
    request.mBandList = {12, 11, 4};
  } else {
    request.mBandList = {4, 3, 2};
  }

  request.mFormat = "image/tiff";

  return request;
}

} // namespace csp::visualquery