////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_CSVLOADER_HPP
#define CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_CSVLOADER_HPP

#include <functional>
#include <string>
#include <vector>

namespace csp::atmospheres::models::bruneton::internal {

class CSVLoader {
 public:
  static std::vector<double> readDensity(std::string const& filename, uint32_t& densityCount);
  static std::vector<std::vector<double>> readPhase(
      std::string const& filename, std::vector<double>& wavelengths);
  static std::vector<double> readExtinction(
      std::string const& filename, std::vector<double>& wavelengths);

 private:
  static std::vector<std::string> lineToArray(std::stringstream& ss, char delimiter);
  static void                     readLines(std::string const&                   filename,
                          std::function<void(long lineNumber, std::string line)> lineConsumer);
};

} // namespace csp::atmospheres::models::bruneton::internal

#endif // CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_CSVLOADER_HPP