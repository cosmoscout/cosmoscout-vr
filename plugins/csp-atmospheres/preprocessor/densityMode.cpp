////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "densityMode.hpp"

#include "../../../src/cs-utils/CommandLine.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////
// The particles can be distributed according to various functions.                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class DensityDistributionType { eExponential, eDoubleExponential, eTent };

// Make the DensityDistributionTypes available for JSON deserialization.
NLOHMANN_JSON_SERIALIZE_ENUM(
    DensityDistributionType, {
                          {DensityDistributionType::eExponential, "exponential"},
                          {DensityDistributionType::eDoubleExponential, "doubleExponential"},
                          {DensityDistributionType::eTent, "tent"},
                      })

////////////////////////////////////////////////////////////////////////////////////////////////////
// This type describes a particle density distribution. Depending on the type, the parameters A   //
// and B will store different things.                                                             //
////////////////////////////////////////////////////////////////////////////////////////////////////

struct DensityDistribution {

  // Type of the distribution, e.g. exponential or double-exponential.
  DensityDistributionType type;

  // For the exponential fall-offs, this is the scale height. For the tent type, this is the
  // altitude of maximum density.
  double paramA;

  // This is only used for the tent type. It specifies the total width of the tent distribution.
  double paramB;

  // This is used in multi-modal distributions to compute the weight of this mode.
  double relativeNumberDensity;
};

// Make this type available for JSON deserialization.
void from_json(nlohmann::json const& j, DensityDistribution& s) {
  j.at("type").get_to(s.type);
  j.at("relativeNumberDensity").get_to(s.relativeNumberDensity);

  switch (s.type) {
  case DensityDistributionType::eExponential:
    j.at("scaleHeight").get_to(s.paramA);
    break;
  case DensityDistributionType::eDoubleExponential:
    j.at("scaleHeight").get_to(s.paramA);
    break;
  case DensityDistributionType::eTent:
    j.at("peakAltitude").get_to(s.paramA);
    j.at("width").get_to(s.paramB);
    break;
  }
}

// Sample the density distribution at a given altitude.
double sampleDensity(DensityDistribution const& distribution, double altitude) {

  if (distribution.type == DensityDistributionType::eExponential) {

    double scaleHeight = distribution.paramA;
    return std::exp(-altitude / scaleHeight);

  } else if (distribution.type == DensityDistributionType::eDoubleExponential) {

    double scaleHeight = distribution.paramA;
    return std::exp(1.0 - 1.0 / std::exp(-altitude / scaleHeight));

  } else if (distribution.type == DensityDistributionType::eTent) {

    double tentHeight      = distribution.paramA;
    double tentWidth       = distribution.paramB;
    double relativeDensity = 1.0 - std::abs((altitude - tentHeight) / tentWidth);
    return std::max(0.0, relativeDensity);
  }

  return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// This struct describes a multi-modal density distribution.                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////

struct DensitySettings {

  // The density distribution can follow a mixture of various distributions.
  std::vector<DensityDistribution> densityModes;
};

// Make this type available for JSON deserialization.
void from_json(nlohmann::json const& j, DensitySettings& s) {
  j.at("densityModes").get_to(s.densityModes);
}

// Samples the density distribution at evenly spaced altitudes.
std::vector<double> sampleDensities(
    DensitySettings const& settings, int32_t count, double minAltitude, double maxAltitude) {

  std::vector<double> densities(count);

  for (int32_t i(0); i < count; ++i) {
    double altitude    = minAltitude + i * (maxAltitude - minAltitude) / (count - 1.0);
    double density     = 0.0;
    double totalWeight = 0.0;

    for (auto const& mode : settings.densityModes) {
      totalWeight += mode.relativeNumberDensity;
      density += sampleDensity(mode, altitude) * mode.relativeNumberDensity;
    }

    densities[i] = density / totalWeight;
  }

  return densities;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

int densityMode(std::vector<std::string> const& arguments) {

  bool        cPrintHelp       = false;
  std::string cInput           = "";
  std::string cOutput          = "particles";
  std::string cLambdas         = "";
  double      cMinAltitude     = 0.0;
  double      cMaxAltitude     = 80000;
  int32_t     cAltitudeSamples = 1024;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  args.addArgument(
      {"-i", "--input"}, &cInput, "The JSON file with the distribution information (required).");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The density data will be written to <name>_density.csv (default: \"" + cOutput + "\").");
  args.addArgument({"--min-altitude"}, &cMinAltitude,
      "The minimum altitude in m (default: " + std::to_string(cMinAltitude) + ").");
  args.addArgument({"--max-altitude"}, &cMaxAltitude,
      "The maximum altitude in m (default: " + std::to_string(cMaxAltitude) + ").");
  args.addArgument({"--altitude-samples"}, &cAltitudeSamples,
      "The number of altitudes to compute (default: " + std::to_string(cAltitudeSamples) + ").");
  args.addArgument({"-h", "--help"}, &cPrintHelp, "Show this help message.");

  // Then do the actual parsing.
  try {
    args.parse(arguments);
  } catch (std::runtime_error const& e) {
    std::cerr << "Failed to parse command line arguments: " << e.what() << std::endl;
    return 1;
  }

  // When cPrintHelp was set to true, we print a help message and exit.
  if (cPrintHelp) {
    args.printHelp();
    return 0;
  }

  // The distribution information is mandatory.
  if (cInput.empty()) {
    std::cerr << "Please specify a distribution information file with the --input option!"
              << std::endl;
    return 1;
  }

  // Try parsing the particle settings.
  std::ifstream  stream(cInput);
  nlohmann::json json;
  stream >> json;
  DensitySettings densitySettings;
  from_json(json, densitySettings);

  if (densitySettings.densityModes.empty()) {
    std::cerr << "There must be at least one entry in 'densityModes'!" << std::endl;
    return 1;
  }

  // Open the output file for writing and write the CSV header.
  std::ofstream output(cOutput + "_density.csv");
  output << "density" << std::endl;

  // Now write a density value for each altitude.
  auto densities = sampleDensities(densitySettings, cAltitudeSamples, cMinAltitude, cMaxAltitude);
  for (auto density : densities) {
    output << fmt::format("{}", density) << std::endl;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
