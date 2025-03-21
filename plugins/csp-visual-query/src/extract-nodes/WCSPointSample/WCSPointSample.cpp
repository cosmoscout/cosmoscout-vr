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

namespace {
template <typename T>
std::vector<std::vector<T>> getPointSampleData(csl::ogc::GDALReader::Texture const& texture) {

  uint32_t       sampleCount = texture.mWidth * texture.mHeight;
  std::vector<T> textureData(static_cast<T*>(texture.mBuffer),
      static_cast<T*>(texture.mBuffer) + texture.mBands * sampleCount);

  std::vector<std::vector<T>> pointData{};

  for (uint32_t b = 0; b < texture.mBands; ++b) {
    double average = 0.0;
    for (uint32_t x = 0; x < texture.mWidth; ++x) {
      for (uint32_t y = 0; y < texture.mHeight; ++y) {
        average += textureData[b * sampleCount + x * texture.mHeight + y];
      }
    }
    pointData.emplace_back(std::vector{static_cast<T>(average / sampleCount)});
  }

  return pointData;
}
} // namespace

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
      std::cout << "UInt8" << std::endl;
      image.mPoints = getPointSampleData<uint8_t>(texture);
      break;
    }

    case 2: // UInt16
    {
      image.mPoints = getPointSampleData<uint16_t>(texture);
      break;
    }

    case 3: // Int16
    {
      image.mPoints = getPointSampleData<int16_t>(texture);
      break;
    }

    case 4: // UInt32
    {
      image.mPoints = getPointSampleData<uint32_t>(texture);
      break;
    }

    case 5: // Int32
    {
      image.mPoints = getPointSampleData<int32_t>(texture);
      break;
    }

    case 6: // Float32
    case 7: {
      image.mPoints = getPointSampleData<float>(texture);
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

  // Single-Pixel requests seem to be impossible, they always return zero data. Hence we do a 2x2
  // request and take the average of the four pixels. The bounding box can be pretty small it seems.
  csl::ogc::Bounds2D bound;
  request.mBounds.mMinLon = coords.first - 0.0001;
  request.mBounds.mMaxLon = coords.first + 0.0001;
  request.mBounds.mMinLat = coords.second - 0.0001;
  request.mBounds.mMaxLat = coords.second + 0.0001;

  request.mMaxSize         = 2;
  request.mKeepAspectRatio = false;
  request.mBandRange       = std::make_pair(1, 255);

  request.mFormat = "image/tiff";

  return request;
}

} // namespace csp::visualquery