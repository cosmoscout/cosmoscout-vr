//
// Created by Tim on 11.05.2022.
//

#ifndef CSP_ATMOSPHERES_MODELS_SCHNEEGANS_INTERNAL_CSVLOADER_HPP
#define CSP_ATMOSPHERES_MODELS_SCHNEEGANS_INTERNAL_CSVLOADER_HPP

#include "Model.hpp"

#include <cassert>
#include <iostream>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>

namespace csp::atmospheres::models::schneegans::internal {

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

} // namespace csp::atmospheres::models::schneegans::internal

#endif // CSP_ATMOSPHERES_MODELS_SCHNEEGANS_INTERNAL_CSVLOADER_HPP