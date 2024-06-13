////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "iorMode.hpp"

#include "common.hpp"
#include "densityMode.hpp"

#include "../../../src/cs-utils/CommandLine.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////
// These gases are currently supported.                                                           //
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class GasType { eNitrogen, eOxygen, eArgon, eCarbonDioxide };

// Make the DensityDistributionTypes available for JSON deserialization.
NLOHMANN_JSON_SERIALIZE_ENUM(GasType, {
                                          {GasType::eNitrogen, "nitrogen"},
                                          {GasType::eOxygen, "oxygen"},
                                          {GasType::eArgon, "argon"},
                                          {GasType::eCarbonDioxide, "carbonDioxide"},
                                      })

////////////////////////////////////////////////////////////////////////////////////////////////////
// Each atmosphere is composed of a number of gases. Each gas is described by its type and the    //
// volume mixing ratio.                                                                           //
////////////////////////////////////////////////////////////////////////////////////////////////////

struct IorComponent {
  GasType type;
  double  volumeMixingRatio;
};

struct IorSettings {
  std::vector<IorComponent> components;
};

void from_json(nlohmann::json const& j, IorComponent& s) {
  j.at("type").get_to(s.type);
  j.at("volumeMixingRatio").get_to(s.volumeMixingRatio);
}

// Make this type available for JSON deserialization.
void from_json(nlohmann::json const& j, IorSettings& s) {
  j.at("components").get_to(s.components);
}

// Computes the refractive index of a gas at a given wavelength (in m), pressure (in Pa), and
// temperature (in K).
double getIoR(GasType type, double lambda, double pressure, double temperature) {
  double mu2 = std::pow(lambda * 1e6, -2.0);
  double n0  = 1.0;

  switch (type) {
    // E. R. Peck and B. N. Khanna. Dispersion of nitrogen
  case GasType::eNitrogen:
    n0 = 1 + 6.8552e-5 + 3.243157e-2 / (144 - mu2);
    break;

    //  J. Zhang, Z. H. Lu, and L. J. Wang. Precision refractive index measurements of air, N2, O2,
    //  Ar, and CO2 with a frequency comb. This is given at 20°C.
  case GasType::eOxygen:
    n0 = 1 + 1.181494e-4 + 9.708931e-3 / (75.4 - mu2);
    n0 = 1 + (n0 - 1) * 293.15 / 273; // Correct for the actual temperature.
    break;

    // E. R. Peck and D. J. Fisher. Dispersion of argon
  case GasType::eArgon:
    n0 = 1 + 6.7867e-5 + 3.0182943e-2 / (144 - mu2);
    break;

    // J. G. Old, K. L. Gentili, and E. R. Peck. Dispersion of Carbon Dioxide
  case GasType::eCarbonDioxide:
    n0 = 1 + 0.00000154489 / (0.0584738 - mu2) + 0.083091927 / (210.9241 - mu2) +
         0.0028764190 / (60.122959 - mu2);
    break;
  }

  // The pressure is given in Pa, the temperature in K. The IoR above is measured at STP (273 K,
  // 101325 Pa). We need to correct for the actual pressure and temperature.
  return 1 + (n0 - 1) * pressure / temperature * 273 / 101325.0;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

int iorMode(std::vector<std::string> const& arguments) {

  bool        cPrintHelp       = false;
  std::string cInput           = "";
  std::string cOutput          = "ior";
  double      cPressure        = 101325;
  double      cTemperature     = 273;
  std::string cLambdas         = "";
  double      cMinLambda       = 0.36e-6;
  double      cMaxLambda       = 0.83e-6;
  int32_t     cLambdaSamples   = 15;
  double      cMinAltitude     = 0.0;
  double      cMaxAltitude     = 80000;
  int32_t     cAltitudeSamples = 1024;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  args.addArgument(
      {"-i", "--input"}, &cInput, "The JSON file with the IoR information (required).");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The IoR data will be written to <name>.csv (default: \"" + cOutput + "\").");
  args.addArgument({"--pressure"}, &cPressure,
      "The pressure in Pa (default: " + std::to_string(cPressure) + ").");
  args.addArgument({"--temperature"}, &cTemperature,
      "The temperature in K (default: " + std::to_string(cTemperature) + ").");
  args.addArgument({"--min-altitude"}, &cMinAltitude,
      "The minimum altitude in m (default: " + std::to_string(cMinAltitude) + ").");
  args.addArgument({"--max-altitude"}, &cMaxAltitude,
      "The maximum altitude in m (default: " + std::to_string(cMaxAltitude) + ").");
  args.addArgument({"--altitude-samples"}, &cAltitudeSamples,
      "The number of altitudes to compute (default: " + std::to_string(cAltitudeSamples) + ").");
  common::addLambdaFlags(args, &cLambdas, &cMinLambda, &cMaxLambda, &cLambdaSamples);
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

  // The input information is mandatory.
  if (cInput.empty()) {
    std::cerr << "Please specify an IoR information file with the --ior-input option!" << std::endl;
    return 1;
  }

  // Try parsing the composition settings.
  std::ifstream  stream(cInput);
  nlohmann::json json;
  stream >> json;
  IorSettings iorSettings;
  from_json(json, iorSettings);

  // Compute the total volume mixing ratio for normalization.
  double totalVolume = 0.0;
  for (auto const& component : iorSettings.components) {
    totalVolume += component.volumeMixingRatio;
  }

  // Open the output file for writing and write the CSV header.
  std::ofstream output(cOutput + ".csv");
  output << "lambda,ior" << std::endl;

  // Now assemble a list of wavelengths in m. This is either provided with the --lambda-samples
  // command-line parameter or via the combination of --min-lambda, --max-lambda, and
  // --lambda-samples.
  std::vector<double> lambdas =
      common::computeLambdas(cLambdas, cMinLambda, cMaxLambda, cLambdaSamples);

  // For each wavelength, compute the IoR and write it to the output file.
  for (double lambda : lambdas) {

    double ior = 0.0;

    for (auto const& component : iorSettings.components) {
      ior += component.volumeMixingRatio * getIoR(component.type, lambda, cPressure, cTemperature) /
             totalVolume;
    }

    output << fmt::format("{},{}", lambda, ior) << std::endl;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
