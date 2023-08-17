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

std::vector<std::vector<double>> CSVLoader::read2DTable(std::string const& filename) {
  std::vector<std::vector<double>> result;

  readLines(filename, [&](long lineNumber, std::string line) {
    std::stringstream ss(trim(removeMultiWhitespaces(replaceTabsWithWhitespaces(line))));

    // Skip empty lines and first line.
    if (ss.rdbuf()->in_avail() == 0 || lineNumber == 0) {
      return;
    }

    auto elements = lineToArray(ss, ',');

    std::vector<double> intensities;

    // Remove angle.
    elements.erase(elements.begin());

    for (auto& e : elements) {
      intensities.push_back(std::stof(e));
    }

    result.push_back(intensities);
  });

  return result;
}

std::vector<double> CSVLoader::read1DTable(std::string const& filename) {
  std::vector<double> result;

  readLines(filename, [&](long lineNumber, std::string line) {
    std::stringstream ss(trim(removeMultiWhitespaces(replaceTabsWithWhitespaces(line))));

    // Skip empty lines and first line.
    if (ss.rdbuf()->in_avail() == 0 || lineNumber == 0) {
      return;
    }

    auto elements = lineToArray(ss, ',');

    result.push_back(std::stof(elements[1]));
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
