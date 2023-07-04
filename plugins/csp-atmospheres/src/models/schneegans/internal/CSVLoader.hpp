//
// Created by Tim on 11.05.2022.
//

#ifndef CSP_ATMOSPHERES_MODELS_SCHNEEGANS_INTERNAL_CSVLOADER_HPP
#define CSP_ATMOSPHERES_MODELS_SCHNEEGANS_INTERNAL_CSVLOADER_HPP

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
  struct PhaseFunctionSpectrumDatum {
    float               angle;
    std::vector<double> intensities;
  };

  struct PhaseFunctionSpectrumInfo {
    std::vector<double>                     wavelengths;
    std::vector<PhaseFunctionSpectrumDatum> angles;
  };

  struct ExtinctionSpectrum {
    std::vector<double> wavelengths;
    std::vector<double> Cext;
    std::vector<double> Qext;
    std::vector<double> Csca;
    std::vector<double> Qsca;
  };

  static std::vector<std::string> lineToArray(std::stringstream& ss, char delimiter);

  static void readLines(
      std::string filename, std::function<void(long lineNumber, std::string line)> lineConsumer);

  static void readPhaseFunctionSpectrum(std::string filename, PhaseFunctionSpectrumInfo* result);

  static size_t getIndex(float wavelength, const std::vector<double>& wavelengths);

  static float toFloat(const std::string& value);

  static void readExtinction(std::string filename, ExtinctionSpectrum* result);
};

} // namespace csp::atmospheres::models::schneegans::internal

#endif // CSP_ATMOSPHERES_MODELS_SCHNEEGANS_INTERNAL_CSVLOADER_HPP