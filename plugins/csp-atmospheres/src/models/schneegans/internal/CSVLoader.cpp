//
// Created by Tim on 11.05.2022.
//

#include "CSVLoader.hpp"

namespace csp::atmospheres::models::schneegans::internal {

namespace {

/**
 * https://www.techiedelight.com/trim-string-cpp-remove-leading-trailing-spaces/
 */
std::string ltrim(const std::string& s) {
  return std::regex_replace(s, std::regex("^\\s+"), std::string(""));
}

std::string rtrim(const std::string& s) {
  return std::regex_replace(s, std::regex("\\s+$"), std::string(""));
}

std::string trim(const std::string& s) {
  return ltrim(rtrim(s));
}

/**
 * http://www.cplusplus.com/forum/beginner/185953/
 */
std::string removeMultiWhitespaces(const std::string& s) {
  return std::regex_replace(s, std::regex("\\s{2,}"), std::string(" "));
}

std::string replaceTabsWithWhitespaces(std::string& s) {
  std::replace(std::begin(s), std::end(s), '\t', ' ');
  return s;
}
} // namespace

std::vector<std::string> CSVLoader::lineToArray(std::stringstream& ss, char delimiter) {
  auto result = std::vector<std::string>();

  std::string text;
  while (std::getline(ss, text, delimiter)) {
    result.push_back(text);
  }
  return result;
}

void CSVLoader::readLines(
    std::string filename, std::function<void(long lineNumber, std::string line)> lineConsumer) {
  std::ifstream file(filename);

  if (!file.is_open())
    throw std::runtime_error("Could not open file " + filename);
  if (!file.good())
    return;

  std::string line;
  long        lineNumber = 1;
  while (std::getline(file, line)) {
    lineConsumer(lineNumber++, line);
  }
  file.close();
}

void CSVLoader::readPhaseFunctionSpectrum(std::string filename, PhaseFunctionSpectrumInfo* result) {
  readLines(std::move(filename), [&](long lineNumber, std::string line) {
    std::stringstream ss(trim(removeMultiWhitespaces(replaceTabsWithWhitespaces(line))));

    // empty line
    if (ss.rdbuf()->in_avail() == 0) {
      return;
    }

    auto elements = lineToArray(ss, ',');

    // wavelengths are enumerated in the csv header line
    if (lineNumber == 1) {
      result->wavelengths.clear();
      result->angles.clear();
      elements.erase(elements.begin()); // remove first column ('theta')
      for (const auto& e : elements) {
        auto wavelength = std::stof(e);
        result->wavelengths.emplace_back(wavelength);
      }
      return;
    }

    PhaseFunctionSpectrumDatum functionSpectrumDatum{};
    functionSpectrumDatum.angle = std::stof(elements[0]);
    std::vector<double> intensities;

    elements.erase(elements.begin());
    for (auto& e : elements) {
      auto intensity = std::stof(e);
      intensities.push_back(intensity);
    }
    functionSpectrumDatum.intensities = intensities; //
    result->angles.push_back(functionSpectrumDatum);

    return;
  });
}

size_t CSVLoader::getIndex(float wavelength, const std::vector<double>& wavelengths) {
  for (size_t i = 0; i < wavelengths.size(); i++) {
    if (wavelength < wavelengths[i]) {
      return i;
    }
  }

  return -1;
}

float CSVLoader::toFloat(const std::string& value) {
  auto v = std::stof(value);
  // if (abs(v) < 0.0000000001)
  //    v = 0.f;
  return v;
}

void CSVLoader::readExtinction(std::string filename, ExtinctionSpectrum* result) {
  result->wavelengths.clear();
  result->Cext.clear();
  result->Qext.clear();

  result->Csca.clear();
  result->Qsca.clear();

  readLines(std::move(filename), [&](long lineNumber, std::string line) {
    std::stringstream ss(trim(removeMultiWhitespaces(replaceTabsWithWhitespaces(line))));

    // empty line
    if (ss.rdbuf()->in_avail() == 0) {
      return;
    }

    auto elements = lineToArray(ss, ',');

    // skip header line
    if (lineNumber == 1) {
      return;
    }

    result->wavelengths.push_back(std::stof(elements[0]));

    result->Cext.push_back(toFloat(elements[1]));
    result->Qext.push_back(toFloat(elements[2]));

    result->Csca.push_back(toFloat(elements[3]));
    result->Qsca.push_back(toFloat(elements[4]));
    return;
  });
}
} // namespace csp::atmospheres::models::schneegans::internal
