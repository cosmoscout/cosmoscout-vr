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

float stringToFloat(std::string const& s) {
  try {
    return std::stof(s);
  } catch (std::exception const& e) {
    std::cout << "Failed to convert string '" << s << "' to float! Using 0 instead." << std::endl;
  }

  return 0.f;
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

std::vector<double> CSVLoader::readDensity(std::string const& filename, uint32_t& densityCount) {
  std::vector<double> result;

  bool checkDensities = densityCount != 0;

  readLines(filename, [&](long lineNumber, std::string line) {
    std::stringstream ss(trim(removeMultiWhitespaces(replaceTabsWithWhitespaces(line))));

    // Skip empty lines and first line.
    if (ss.rdbuf()->in_avail() == 0 || lineNumber == 0) {
      return;
    }

    auto elements = lineToArray(ss, ',');
    result.push_back(stringToFloat(elements[0]));
  });

  if (checkDensities) {
    if (densityCount != result.size()) {
      throw std::runtime_error(
          "Failed to read density from '" + filename +
          "': Number of density values differs from a previously loaded data set!");
    }
  }

  densityCount = result.size();

  return result;
}

std::vector<std::vector<double>> CSVLoader::readPhase(
    std::string const& filename, std::vector<double>& wavelengths) {

  std::vector<std::vector<double>> phaseFunctions;

  bool checkWavelengths = !wavelengths.empty();

  readLines(filename, [&](long lineNumber, std::string line) {
    std::stringstream ss(trim(removeMultiWhitespaces(replaceTabsWithWhitespaces(line))));

    // Skip empty lines and first line.
    if (ss.rdbuf()->in_avail() == 0 || lineNumber == 0) {
      return;
    }

    auto elements = lineToArray(ss, ',');

    // Convert to nanometers.
    float lambda = stringToFloat(elements[0]) * 1e9;

    if (checkWavelengths) {
      if (lambda != wavelengths[lineNumber - 1]) {
        throw std::runtime_error(
            "Failed to read phase from '" + filename +
            "': Wavelengths are not the same as in a previously loaded data set!");
      }
    } else {
      wavelengths.push_back(lambda);
    }

    std::vector<double> phase;

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

  std::vector<std::vector<double>> spectra(angleCount);
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

std::vector<double> CSVLoader::readExtinction(
    std::string const& filename, std::vector<double>& wavelengths) {
  std::vector<double> result;

  bool checkWavelengths = !wavelengths.empty();

  readLines(filename, [&](long lineNumber, std::string line) {
    std::stringstream ss(trim(removeMultiWhitespaces(replaceTabsWithWhitespaces(line))));

    // Skip empty lines and first line.
    if (ss.rdbuf()->in_avail() == 0 || lineNumber == 0) {
      return;
    }

    auto elements = lineToArray(ss, ',');

    // Convert to nanometers.
    float lambda = stringToFloat(elements[0]) * 1e9;

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

std::vector<std::string> CSVLoader::lineToArray(std::stringstream& ss, char delimiter) {
  auto result = std::vector<std::string>();

  std::string text;
  while (std::getline(ss, text, delimiter)) {
    result.push_back(text);
  }
  return result;
}

void CSVLoader::readLines(std::string const&               filename,
    std::function<void(long lineNumber, std::string line)> lineConsumer) {
  std::ifstream file(filename);

  if (!file.is_open())
    throw std::runtime_error("Could not open file " + filename);
  if (!file.good())
    return;

  std::string line;
  long        lineNumber = 0;
  while (std::getline(file, line)) {
    lineConsumer(lineNumber++, line);
  }
  file.close();
}

} // namespace csp::atmospheres::models::schneegans::internal
