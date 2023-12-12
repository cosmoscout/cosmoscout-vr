////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "densityMode.hpp"

#include "../../src/cs-utils/CommandLine.hpp"
#include "../../src/cs-utils/utils.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////
// The particles can be distributed according to various functions.                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class DistributionType { eExponential, eDoubleExponential, eTent };

// Make the DistributionTypes available for JSON deserialization.
NLOHMANN_JSON_SERIALIZE_ENUM(
    DistributionType, {
                          {DistributionType::eExponential, "exponential"},
                          {DistributionType::eDoubleExponential, "doubleExponential"},
                          {DistributionType::eTent, "tent"},
                      })

////////////////////////////////////////////////////////////////////////////////////////////////////
// This type describes a particle density distribution. Depending on the type, the parameters A   //
// and B will store different things.                                                             //
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Distribution {

  // Type of the distribution, e.g. exponential or double-exponential.
  DistributionType type;

  // For the exponential fall-offs, this is the molecular number density at sea level. For the tent
  // type, this is the molecular number density at the peak of the tent.
  double numberDensity;

  // For the exponential fall-offs, this is the scale height. For the tent type, this is the
  // altitude of maximum density.
  double paramA;

  // This is only used for the tent type. It specifies the total width of the tent distribution.
  double paramB;
};

// Make this type available for JSON deserialization.
void from_json(nlohmann::json const& j, Distribution& s) {
  j.at("type").get_to(s.type);
  j.at("numberDensity").get_to(s.numberDensity);

  switch (s.type) {
  case DistributionType::eExponential:
    j.at("scaleHeight").get_to(s.paramA);
    break;
  case DistributionType::eDoubleExponential:
    j.at("scaleHeight").get_to(s.paramA);
    break;
  case DistributionType::eTent:
    j.at("peakAltitude").get_to(s.paramA);
    j.at("width").get_to(s.paramB);
    break;
  }
}

// Sample the density distribution at a given altitude.
double sampleDensity(Distribution const& distribution, double altitude) {

  if (distribution.type == DistributionType::eExponential) {

    double scaleHeight = distribution.paramA;
    return distribution.numberDensity * std::exp(-altitude / scaleHeight);

  } else if (distribution.type == DistributionType::eDoubleExponential) {

    double scaleHeight = distribution.paramA;
    return distribution.numberDensity * std::exp(1.0 - 1.0 / std::exp(-altitude / scaleHeight));

  } else if (distribution.type == DistributionType::eTent) {

    double tentHeight      = distribution.paramA;
    double tentWidth       = distribution.paramB;
    double relativeDensity = 1.0 - std::abs((altitude - tentHeight) / tentWidth);
    return distribution.numberDensity * std::max(0.0, relativeDensity);
  }

  return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// This struct describes a multi-modal density distribution.                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////

struct DensitySettings {

  // The density distribution can follow a mixture of various distributions.
  std::vector<Distribution> densityModes;
};

// Make this type available for JSON deserialization.
void from_json(nlohmann::json const& j, DensitySettings& s) {
  j.at("densityModes").get_to(s.densityModes);
}

// Samples the density distribution at evenly spaced altitudes.
std::map<double, double> sampleDensities(
    DensitySettings const& settings, int32_t count, double minAltitude, double maxAltitude) {

  std::map<double, double> densities;

  for (int32_t i(0); i < count; ++i) {
    double altitude = minAltitude + i * (maxAltitude - minAltitude) / (count - 1.0);
    double density  = 0.0;

    for (auto const& mode : settings.densityModes) {
      density += sampleDensity(mode, altitude);
    }

    densities[altitude] = density;
  }

  return densities;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

int densityMode(std::vector<std::string> const& arguments) {

  bool        cPrintHelp       = false;
  std::string cInput           = "";
  std::string cOutput          = "densities.csv";
  std::string cLambdas         = "";
  double      cMinAltitude     = 0.0;
  double      cMaxAltitude     = 80000;
  int32_t     cAltitudeSamples = 1024;

  // First configure all possible command line options.
  cs::utils::CommandLine args(
      "Welcome to the density preprocessor! Here are the available options:");
  args.addArgument(
      {"-i", "--input"}, &cInput, "The JSON file with the distribution information (required).");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The density data will be written to this CSV file (default: \"" + cOutput + "\").");
  args.addArgument({"--min-altitude"}, &cMinAltitude,
      "The minimum wavelength in µm (default: " + std::to_string(cMinAltitude) + ").");
  args.addArgument({"--max-altitude"}, &cMaxAltitude,
      "The maximum wavelength in µm (default: " + std::to_string(cMaxAltitude) + ").");
  args.addArgument({"--altitude-samples"}, &cAltitudeSamples,
      "The number of wavelengths to compute (default: " + std::to_string(cAltitudeSamples) + ").");
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
  std::ofstream output(cOutput);
  output << "altitude,numberDensity" << std::endl;

  // Now write a density value for each altitude.
  auto densities = sampleDensities(densitySettings, cAltitudeSamples, cMinAltitude, cMaxAltitude);
  for (auto [altitude, density] : densities) {
    output << fmt::format("{},{}", altitude, density) << std::endl;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
