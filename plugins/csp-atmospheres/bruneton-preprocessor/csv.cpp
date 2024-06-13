////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "csv.hpp"

#include <fstream>
#include <iostream>
#include <regex>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper methods to trim left and right whitespace from a string. Based on:
// https://www.techiedelight.com/trim-string-cpp-remove-leading-trailing-spaces/
std::string ltrim(const std::string& s) {
  return std::regex_replace(s, std::regex("^\\s+"), std::string(""));
}

std::string rtrim(const std::string& s) {
  return std::regex_replace(s, std::regex("\\s+$"), std::string(""));
}

std::string trim(const std::string& s) {
  return ltrim(rtrim(s));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper method to remove consecutive whitespaces. Based on:
// http://www.cplusplus.com/forum/beginner/185953/
std::string removeMultiWhitespaces(const std::string& s) {
  return std::regex_replace(s, std::regex("\\s{2,}"), std::string(" "));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper method to replace tabs by spaces.
std::string replaceTabsWithWhitespaces(std::string& s) {
  std::replace(std::begin(s), std::end(s), '\t', ' ');
  return s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Safe conversion of strings to floats.
float stringToFloat(std::string const& s) {
  try {
    return std::stof(s);
  } catch (std::exception const&) {
    std::cout << "Failed to convert string '" << s << "' to float! Using 0 instead." << std::endl;
  }

  return 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper method to read a line of text from an input stream and split it at the given delimiters.
std::vector<std::string> lineToArray(std::stringstream& ss, char delimiter) {
  auto result = std::vector<std::string>();

  std::string text;
  while (std::getline(ss, text, delimiter)) {
    result.push_back(text);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Reads the given file line by line and calls the given callback for each line.
void readLines(std::string const&                          filename,
    std::function<void(long lineNumber, std::string line)> lineConsumer) {
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Could not open file " + filename);
  }

  if (!file.good()) {
    return;
  }

  std::string line;
  long        lineNumber = 0;
  while (std::getline(file, line)) {
    lineConsumer(lineNumber++, line);
  }
  file.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csv {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> readDensity(std::string const& filename, uint32_t& densityCount) {
  std::vector<float> result;

  // Read the file line-by-line.
  readLines(filename, [&](long lineNumber, std::string line) {
    std::stringstream ss(trim(removeMultiWhitespaces(replaceTabsWithWhitespaces(line))));

    // Skip empty lines and first line.
    if (ss.rdbuf()->in_avail() == 0 || lineNumber == 0) {
      return;
    }

    auto elements = lineToArray(ss, ',');
    result.push_back(stringToFloat(elements[0]));
  });

  // We compare the number of read densities to the target number if given.
  if (densityCount != 0) {
    if (densityCount != result.size()) {
      throw std::runtime_error(
          "Failed to read density from '" + filename +
          "': Number of density values differs from a previously loaded data set!");
    }
  } else {
    densityCount = static_cast<uint32_t>(result.size());
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<float>> readPhase(
    std::string const& filename, std::vector<float>& wavelengths) {

  std::vector<std::vector<float>> phaseFunctions;

  // We will check the read wavelengths if some target wavelengths are given.
  bool checkWavelengths = !wavelengths.empty();

  // Read the file line-by-line.
  readLines(filename, [&](long lineNumber, std::string line) {
    std::stringstream ss(trim(removeMultiWhitespaces(replaceTabsWithWhitespaces(line))));

    // Skip empty lines and first line.
    if (ss.rdbuf()->in_avail() == 0 || lineNumber == 0) {
      return;
    }

    auto elements = lineToArray(ss, ',');

    // Convert to nanometers.
    float lambda = stringToFloat(elements[0]) * 1e9F;

    if (checkWavelengths) {
      if (lambda != wavelengths[lineNumber - 1]) {
        throw std::runtime_error(
            "Failed to read phase from '" + filename +
            "': Wavelengths are not the same as in a previously loaded data set!");
      }
    } else {
      wavelengths.push_back(lambda);
    }

    std::vector<float> phase;

    // Remove first lambda column.
    elements.erase(elements.begin());

    for (auto& e : elements) {
      phase.push_back(stringToFloat(e));
    }

    phaseFunctions.push_back(phase);
  });

  // Now we need to transpose this matrix as we do not want to return a phase function for each
  // lambda but a spectrum for each phase function angle.
  size_t angleCount  = phaseFunctions.front().size();
  size_t lambdaCount = wavelengths.size();

  std::vector<std::vector<float>> spectra(angleCount);
  for (size_t i(0); i < angleCount; ++i) {
    spectra[i].resize(lambdaCount);
  }

  for (size_t i(0); i < lambdaCount; ++i) {
    for (size_t j(0); j < angleCount; ++j) {
      spectra[j][i] = phaseFunctions[i][j];
    }
  }

  return spectra;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> readExtinction(std::string const& filename, std::vector<float>& wavelengths) {
  std::vector<float> result;

  // We will check the read wavelengths if some target wavelengths are given.
  bool checkWavelengths = !wavelengths.empty();

  // Read the file line-by-line.
  readLines(filename, [&](long lineNumber, std::string line) {
    std::stringstream ss(trim(removeMultiWhitespaces(replaceTabsWithWhitespaces(line))));

    // Skip empty lines and first line.
    if (ss.rdbuf()->in_avail() == 0 || lineNumber == 0) {
      return;
    }

    auto elements = lineToArray(ss, ',');

    // Convert to nanometers.
    float lambda = stringToFloat(elements[0]) * 1e9F;

    if (checkWavelengths) {
      if (lambda != wavelengths[lineNumber - 1]) {
        throw std::runtime_error(
            "Failed to read extinction from '" + filename +
            "': Wavelengths are not the same as in a previously loaded data set!");
      }
    } else {
      wavelengths.push_back(lambda);
    }

    result.push_back(stringToFloat(elements[1]));
  });

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> readIoR(std::string const& filename, std::vector<float>& wavelengths) {
  std::vector<float> result;

  // We will check the read wavelengths if some target wavelengths are given.
  bool checkWavelengths = !wavelengths.empty();

  // Read the file line-by-line.
  readLines(filename, [&](long lineNumber, std::string line) {
    std::stringstream ss(trim(removeMultiWhitespaces(replaceTabsWithWhitespaces(line))));

    // Skip empty lines and first line.
    if (ss.rdbuf()->in_avail() == 0 || lineNumber == 0) {
      return;
    }

    auto elements = lineToArray(ss, ',');

    // Convert to nanometers.
    float lambda = stringToFloat(elements[0]) * 1e9F;

    if (checkWavelengths) {
      if (lambda != wavelengths[lineNumber - 1]) {
        throw std::runtime_error(
            "Failed to read IoR data from '" + filename +
            "': Wavelengths are not the same as in a previously loaded data set!");
      }
    } else {
      wavelengths.push_back(lambda);
    }

    result.push_back(stringToFloat(elements[1]));
  });

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csv
