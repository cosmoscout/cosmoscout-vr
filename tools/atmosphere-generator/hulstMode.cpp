////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "hulstMode.hpp"

#include "common.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <complex>
#include <fstream>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
double anomalousDiffraction(double wavelength, double radius, std::complex<double> n) {
  double x       = 2.0 * glm::pi<double>() * radius / wavelength;
  double rho     = 2.0 * x * (n.real() - 1.0);
  double tanBeta = -n.imag() / (n.real() - 1.0);
  double beta    = std::atan(tanBeta);

  double q_ext = 2.0;
  q_ext -= 4.0 * std::pow(glm::e<double>(), -rho * tanBeta) * (std::cos(beta) / rho) *
           std::sin(rho - beta);
  q_ext -= 4.0 * std::pow(glm::e<double>(), -rho * tanBeta) * std::pow(std::cos(beta) / rho, 2.0) *
           std::cos(rho - 2.0 * beta);
  q_ext += 4.0 * std::pow(std::cos(beta) / rho, 2.0) * std::cos(2.0 * beta);

  return q_ext;
}

// https://link.springer.com/content/pdf/10.1186/s41074-016-0012-1.pdf
// https://github.com/OpenSpace/OpenSpace/blob/integration/paper-atmosphere/modules/atmosphere/shaders/atmosphere_common.glsl#L328
double hulstScattering(double wavelength, double turbidity, double kappa, double jungeExponent) {
  double c        = (0.65 * turbidity - 0.65) * 1e-16;
  double beta_sca = 0.434 * c * glm::pi<double>() *
                    std::pow(2.0 * glm::pi<double>() / wavelength, jungeExponent - 2.0) * kappa;
  return beta_sca;
}
} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

int hulstMode(std::vector<std::string> const& arguments) {

  bool        cPrintHelp     = false;
  std::string cOutput        = "hulst";
  double      cJunge         = 4.0;
  double      cTurbidity     = 1.007;
  std::string cKappa         = "0.15";
  double      cRadius        = 1.6e-6;
  std::string cIoRreal       = "1.52";
  std::string cIoRImag       = "0.0";
  double      cNumberDensity = 0.02e6;
  std::string cLambdas       = "";
  double      cMinLambda     = 0.36e-6;
  double      cMaxLambda     = 0.83e-6;
  int32_t     cLambdaSamples = 15;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The scattering data will be written to <name>_scattering.csv and "
      "<name>_absorption.csv, respectively (default: \"" +
          cOutput + "\").");
  args.addArgument({"--junge"}, &cJunge,
      fmt::format("The Junge exponent for the scattering approximation (default: {}).", cJunge));
  args.addArgument({"--turbidity"}, &cTurbidity,
      fmt::format(
          "The Junge exponent for the scattering approximation (default: {}).", cTurbidity));
  args.addArgument({"--kappa"}, &cKappa,
      fmt::format("The Kappa factor for the scattering approximation. This can be a single value "
                  "or a comma-separated list of values per wavelength (default: {}).",
          cKappa));
  args.addArgument({"-r", "--radius"}, &cRadius,
      fmt::format("The particle radius for the anomalous diffraction approximation (default: {}).",
          cRadius));
  args.addArgument({"-n", "--ior-real"}, &cIoRreal,
      fmt::format("The real part of the particle's IoR for the anomalous diffraction "
                  "approximation. This can be a single value or a comma-separated list of values "
                  "per wavelength  (default: {}).",
          cIoRreal));
  args.addArgument({"-k", "--ior-imag"}, &cIoRImag,
      fmt::format("The imaginary part of the particle's IoR for the anomalous diffraction "
                  "approximation. This can be a single value or a comma-separated list of values "
                  "per wavelength  (default: {}).",
          cIoRImag));
  args.addArgument({"--number-density"}, &cNumberDensity,
      fmt::format("The number of particle per unit volume to compute the scattering coefficients "
                  "(default: {}).",
          cNumberDensity));
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

  // Now assemble a list of wavelengths in m. This is either provided with the --lambda-samples
  // command-line parameter or via the combination of --min-lambda, --max-lambda, and
  // --lambda-samples.
  std::vector<double> lambdas =
      common::computeLambdas(cLambdas, cMinLambda, cMaxLambda, cLambdaSamples);

  if (lambdas.empty()) {
    return 1;
  }

  std::vector<double> ior_reals = common::parseNumberList(cIoRreal);
  if (ior_reals.size() != 1 && ior_reals.size() != lambdas.size()) {
    std::cerr << "There must be either one real part of the IoR or one per wavelength!"
              << std::endl;
    return 1;
  }

  std::vector<double> ior_imags = common::parseNumberList(cIoRImag);
  if (ior_imags.size() != 1 && ior_imags.size() != lambdas.size()) {
    std::cerr << "There must be either one imaginary part of the IoR or one per wavelength!"
              << std::endl;
    return 1;
  }

  std::vector<double> kappas = common::parseNumberList(cKappa);
  if (kappas.size() != 1 && kappas.size() != lambdas.size()) {
    std::cerr << "There must be either one kappa or one per wavelength!" << std::endl;
    return 1;
  }

  // Open the output files and write the CSV headers.
  std::ofstream scatteringOutput(cOutput + "_scattering.csv");
  std::ofstream absorptionOutput(cOutput + "_absorption.csv");

  scatteringOutput << "lambda,beta_sca" << std::endl;
  absorptionOutput << "lambda,beta_abs" << std::endl;

  // Now write a line to the CSV file for each wavelength.
  for (size_t i(0); i < lambdas.size(); ++i) {
    double lambda = lambdas[i];

    double ior_real = ior_reals.size() == 1 ? ior_reals[0] : ior_reals[i];
    double ior_imag = ior_imags.size() == 1 ? ior_imags[0] : ior_imags[i];
    double kappa    = kappas.size() == 1 ? kappas[0] : kappas[i];

    double q_ext = anomalousDiffraction(lambda, cRadius, std::complex<double>(ior_real, ior_imag));
    double c_ext = q_ext * glm::pi<double>() * cRadius * cRadius;
    double beta_ext = c_ext * cNumberDensity;

    double beta_sca = hulstScattering(lambda, cTurbidity, kappa, cJunge);
    double beta_abs = beta_ext - beta_sca;

    scatteringOutput << fmt::format("{},{}", lambda, beta_sca) << std::endl;
    absorptionOutput << fmt::format("{},{}", lambda, beta_abs) << std::endl;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
