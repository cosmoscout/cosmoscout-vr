////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../../../../src/cs-core/Settings.hpp"
#include "../../../../src/cs-utils/filesystem.hpp"
#include "TransferFunction.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

// testing:
#include <fstream>

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string TransferFunction::sName = "TransferFunction";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string TransferFunction::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/TransferFunction.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<TransferFunction> TransferFunction::sCreate() {
  return std::make_unique<TransferFunction>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& TransferFunction::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TransferFunction::process() {
  /*
  auto minMax = readInput<std::pair<double, double>>("minMax", std::pair<double,double>(0, 256));

  TODO:
  - statistics node (min, max, average)
  - WCS min/max aus anfrage?
  - min/max input in renderer
  - min/max in transfer ohne fkt


  if (minMax.first != 0 || minMax.second != 256) {
    // scale the lut to the difference of the min and max value
    const int diff = static_cast<const int>(std::abs(minMax.second - minMax.first));
    std::vector<glm::vec4> scaledLUT(diff);

    for (int i = 0; i < diff; ++i) {

      // get current position in scaledLUT
      double scaledPos = static_cast<double>(i) / diff;
      
      // get closest indicies to scaledLUT pos in mLUT
      double mLutPos = scaledPos * 255;
      int lowerIndex = static_cast<int>(std::floor(mLutPos));
      int upperIndex = static_cast<int>(std::ceil(mLutPos));

      // ratio between lowerIndex and upperIndex
      double t = mLutPos - lowerIndex;

      // lerp both indicies
      scaledLUT[i] = lerpVec4(mLut[lowerIndex], mLut[upperIndex], t);
    }

    {
      nlohmann::json ogLUTJson;
      ogLUTJson["lut"] = mLut;
      std::ofstream file("C:/Users/sass_fl/ogLUT.json");
      file << ogLUTJson;

      nlohmann::json scaledLUTJson;
      scaledLUTJson["lut"] = scaledLUT;
      std::ofstream file2("C:/Users/sass_fl/scaledLUT.json");
      file2 << scaledLUTJson;
    }

    mLut = scaledLUT;
  }*/

  std::cout << "lut size: " << mLut.size() << std::endl;
  writeOutput("lut", mLut);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec4 TransferFunction::lerpVec4(glm::vec4 v0, glm::vec4 v1, double t) {
  // TODO: lerp colors via HSV and not RGB
  return glm::vec4(
    v0[0] + t * (v1[0] - v0[0]),
    v0[1] + t * (v1[1] - v0[1]),
    v0[2] + t * (v1[2] - v0[2]),
    v0[3] + t * (v1[3] - v0[3])
  );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TransferFunction::onMessageFromJS(nlohmann::json const& message) {
  if (message.find("lut") == message.end()) {
    return;
  }

  cs::core::Settings::deserialize(message, "lut", mLut);
  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json TransferFunction::getData() const {
  return {}; // TODO
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TransferFunction::setData(nlohmann::json const& json) {
  // TODO
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
