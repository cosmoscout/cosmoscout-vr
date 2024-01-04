////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "rayleighMode.hpp"

#include "common.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

////////////////////////////////////////////////////////////////////////////////////////////////////

// Values from Table III in Penndorf 1957 "Tables of the Refractive Index for
// Standard Air and the Rayleigh Scattering Coefficient for the Spectral
// Region between 0.2 and 20.0 μ and Their Application to Atmospheric Optics".
static const float PENNDORF[48] = {70.45E-6, 62.82E-6, 56.20E-6, 50.43E-6, 45.40E-6, 40.98E-6,
    37.08E-6, 33.65E-6, 30.60E-6, 27.89E-6, 25.48E-6, 23.33E-6, 21.40E-6, 19.66E-6, 18.10E-6,
    16.69E-6, 15.42E-6, 14.26E-6, 13.21E-6, 12.26E-6, 11.39E-6, 10.60E-6, 9.876E-6, 9.212E-6,
    8.604E-6, 8.045E-6, 7.531E-6, 7.057E-6, 6.620E-6, 6.217E-6, 5.844E-6, 5.498E-6, 5.178E-6,
    4.881E-6, 4.605E-6, 4.348E-6, 4.109E-6, 3.886E-6, 3.678E-6, 3.484E-6, 3.302E-6, 3.132E-6,
    2.973E-6, 2.824E-6, 2.684E-6, 2.583E-6, 2.481E-6, 2.380E-6};

////////////////////////////////////////////////////////////////////////////////////////////////////

int rayleighMode(std::vector<std::string> const& arguments) {

  bool        cPrintHelp                = false;
  std::string cOutput                   = "rayleigh";
  bool        cPenndorfPhase            = false;
  bool        cPenndorfExtinction       = false;
  std::string cIoR                      = "1.00028276";
  double      cNumberDensity            = 2.68731e25;
  double      cPhaseDepolarization      = 0.0;
  double      cScatteringDepolarization = 0.0;
  std::string cLambdas                  = "";
  double      cMinLambda                = 0.36e-6;
  double      cMaxLambda                = 0.83e-6;
  int32_t     cLambdaSamples            = 15;
  int32_t     cThetaSamples             = 91;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The scattering data will be written to <name>_phase.csv, <name>_scattering.csv, and "
      "<name>_absorption.csv, respectively (default: \"" +
          cOutput + "\").");
  args.addArgument(
      {"--penndorf-phase"}, &cPenndorfPhase, "Use the Penndorf phase function (default: false)");
  args.addArgument({"--penndorf-extinction"}, &cPenndorfExtinction,
      "Use the Penndorf extinction tables (default: false)");
  args.addArgument({"--ior"}, &cIoR,
      fmt::format("The index of refraction of the gas. Can be one number or a comma-separated list "
                  "of values for each wavelength (default: {}).",
          cIoR));
  args.addArgument({"-n", "--number-density"}, &cNumberDensity,
      fmt::format("The number density per m³ (default: {}).", cNumberDensity));
  args.addArgument({"--phase-depolarization"}, &cPhaseDepolarization,
      fmt::format(
          "The depolarization factor for the phase function (default: {}).", cPhaseDepolarization));
  args.addArgument({"--scattering-depolarization"}, &cScatteringDepolarization,
      fmt::format("The depolarization factor for the king-correction used for the scattering "
                  "coefficient (default: {}).",
          cScatteringDepolarization));
  common::addLambdaFlags(args, &cLambdas, &cMinLambda, &cMaxLambda, &cLambdaSamples);
  common::addThetaFlags(args, &cThetaSamples);
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

  // Now assemble a list of wavelengths in m. This is either provided with the --lambda-samples
  // command-line parameter or via the combination of --min-lambda, --max-lambda, and
  // --lambda-samples.
  std::vector<double> lambdas =
      common::computeLambdas(cLambdas, cMinLambda, cMaxLambda, cLambdaSamples);

  if (lambdas.empty()) {
    return 1;
  }

  std::vector<double> iors = common::parseNumberList(cIoR);

  if (iors.size() != 1 && iors.size() != lambdas.size()) {
    std::cerr << "There must be either one index of refraction or one per wavelength!" << std::endl;
    return 1;
  }

  // We will write this many phase function samples. cThetaSamples determines the number of samples
  // between 0° and 90° (including both).
  int32_t totalAngles = cThetaSamples * 2 - 1;

  // Open the output files and write the CSV headers.
  std::ofstream phaseOutput(cOutput + "_phase.csv");
  std::ofstream scatteringOutput(cOutput + "_scattering.csv");
  std::ofstream absorptionOutput(cOutput + "_absorption.csv");

  scatteringOutput << "lambda,beta_sca" << std::endl;
  absorptionOutput << "lambda,beta_abs" << std::endl;
  phaseOutput << "lambda";
  for (int32_t t(0); t < totalAngles; ++t) {
    phaseOutput << fmt::format(",{}", 180.0 * t / (totalAngles - 1.0));
  }
  phaseOutput << std::endl;

  // Now write a line to the CSV file for each wavelength.
  for (size_t i(0); i < lambdas.size(); ++i) {
    double lambda = lambdas[i];

    // Print scattering coefficient.
    if (cPenndorfExtinction) {

      std::vector<double> penndorf;
      for (int j = 0; j < 48; ++j) {
        // The above values are for T_0=0°C. For T=15°C, a correction factor
        // T_0 / T must be applied (Eq. (12) in Penndorf paper).
        constexpr double T_0 = 273.16;
        constexpr double T   = T_0 + 15.0;
        penndorf.push_back(PENNDORF[j] * (T_0 / T));
      }

      const double minLambda = 0.36e-6;
      const double maxLambda = 0.83e-6;

      double beta_sca = common::interpolate(penndorf, minLambda, maxLambda, lambda);

      scatteringOutput << fmt::format("{},{}", lambda, beta_sca) << std::endl;

    } else {

      double ior = iors.size() == 1 ? iors[0] : iors[i];

      double f = std::pow((ior * ior - 1.0), 2.0) * (6.0 + 3.0 * cScatteringDepolarization) /
                 (6.0 - 7.0 * cScatteringDepolarization);
      double beta_sca =
          8.0 / 3.0 * std::pow(glm::pi<double>(), 3.0) * f / (cNumberDensity * std::pow(lambda, 4));

      scatteringOutput << fmt::format("{},{}", lambda, beta_sca) << std::endl;
    }

    // Absorption is always zero.
    absorptionOutput << fmt::format("{},{}", lambda, 0.0) << std::endl;

    phaseOutput << fmt::format("{}", lambda);
    for (int32_t i(0); i < totalAngles; ++i) {
      double theta = i * glm::pi<double>() / (totalAngles - 1);
      double gamma = cPhaseDepolarization / (2.0 - cPhaseDepolarization);
      double phase =
          cPenndorfPhase
              ? 0.7629 / (4.0 * glm::pi<double>()) * (1.0 + 0.932 * std::pow(std::cos(theta), 2.0))
              : 3.0 / (16.0 * glm::pi<double>()) *
                    (1.0 + 3.0 * gamma + (1.0 - gamma) * std::pow(std::cos(theta), 2.0)) /
                    (1.0 + 2.0 * gamma);
      phaseOutput << fmt::format(",{}", phase);
    }
    phaseOutput << std::endl;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
