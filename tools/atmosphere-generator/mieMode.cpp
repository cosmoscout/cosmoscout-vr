////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../../src/cs-utils/CommandLine.hpp"

#include <nlohmann/json.hpp>

#include <complex>
#include <fstream>
#include <map>

// Mie Mode:
//   Input:
//      samplingSettings:
//        minLambda
//        maxLambda
//        lambdaSteps
//        thetaSteps
//      particleSettings:
//        sizeDistribution
//        refractiveIndex
//   Output:
//     phase function for each lambda
//     betaSca for each lambda
//     betaAbs for each lambda
//
// Rayleigh Mode:
//   Input:
//      samplingSettings:
//        minLambda
//        maxLambda
//        lambdaSteps
//        thetaSteps
//      particleSettings:
//        sizeDistribution
//        refractiveIndex
//   Output:
//     phase function for each lambda
//     betaSca for each lambda
//     betaAbs for each lambda
//
// Ozone Mode:
//   Input:
//      samplingSettings:
//        minLambda
//        maxLambda
//        lambdaSteps
//   Output:
//     betaAbs for each lambda
//

enum class DistributionType { eGamma, eLogNormal };
typedef std::map<double, std::complex<double>> IoRSpectrum;

struct SamplingSettings {
  int32_t minLambda   = 360;
  int32_t maxLambda   = 830;
  int32_t lambdaSteps = 15;
  int32_t thetaSteps  = 1024;
};

struct Distribution {
  DistributionType sizeDistribution;
  double           effectiveRadius;
  double           effectiveVariance;
  double           relativeAmount;
};

struct ParticleInclusion {
  double      fraction;
  IoRSpectrum ior;
};

struct ParticleSettings {
  std::vector<Distribution>        sizeModes;
  IoRSpectrum                      ior;
  std::optional<ParticleInclusion> inclusion;
};

namespace nlohmann {
template <typename T>
struct adl_serializer<std::complex<T>> {
  static void from_json(json const& j, std::complex<T>& opt) {
    T real, imag;
    j.at(0).get_to(real);
    j.at(1).get_to(imag);
    opt.real(real);
    opt.imag(imag);
  }
};
} // namespace nlohmann

NLOHMANN_JSON_SERIALIZE_ENUM(DistributionType, {
                                                   {DistributionType::eGamma, "gamma"},
                                                   {DistributionType::eLogNormal, "logNormal"},
                                               })

void from_json(nlohmann::json const& j, Distribution& s) {
  j.at("sizeDistribution").get_to(s.sizeDistribution);
  j.at("effectiveRadius").get_to(s.effectiveRadius);
  j.at("effectiveVariance").get_to(s.effectiveVariance);
  j.at("relativeAmount").get_to(s.relativeAmount);
}

void from_json(nlohmann::json const& j, ParticleInclusion& s) {
  j.at("fraction").get_to(s.fraction);
  j.at("ior").get_to(s.ior);
}

void from_json(nlohmann::json const& j, ParticleSettings& s) {
  j.at("sizeModes").get_to(s.sizeModes);
  j.at("ior").get_to(s.ior);

  if (j.contains("inclusion")) {
    ParticleInclusion inclusion;
    j.at("inclusion").get_to(inclusion);
    s.inclusion = inclusion;
  }
}

int mieMode(std::vector<std::string> const& arguments) {

  SamplingSettings samplingSettings;
  bool             cPrintHelp = false;
  std::string      cInput     = "";
  std::string      cOutput    = "particles.csv";

  // First configure all possible command line options.
  cs::utils::CommandLine args("Welcome to the Mie preprocessor! Here are the available options:");
  args.addArgument(
      {"-i", "--input"}, &cInput, "The JSON file with the particle information (required).");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The image will be written to this CSV file (default: \"" + cOutput + "\").");
  args.addArgument({"--min-lambda"}, &samplingSettings.minLambda,
      "The minimum wavelength in nm (default: " + std::to_string(samplingSettings.minLambda) +
          ").");
  args.addArgument({"--max-lambda"}, &samplingSettings.maxLambda,
      "The maximum wavelength in nm (default: " + std::to_string(samplingSettings.maxLambda) +
          ").");
  args.addArgument({"--lambda-steps"}, &samplingSettings.lambdaSteps,
      "The number of wavelengths to compute (default: " +
          std::to_string(samplingSettings.lambdaSteps) + ").");
  args.addArgument({"--theta-steps"}, &samplingSettings.thetaSteps,
      "The number of angles to compute (default: " + std::to_string(samplingSettings.thetaSteps) +
          ").");
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

  if (cInput.empty()) {
    std::cerr << "Please specify a particle configuration file with the --input option!"
              << std::endl;
    return 1;
  }

  std::ifstream  stream(cInput);
  nlohmann::json json;
  stream >> json;
  ParticleSettings particleSettings;
  from_json(json, particleSettings);

  if (particleSettings.ior.empty()) {
    std::cerr << "There must be at least one entry in 'ior'!" << std::endl;
    return 1;
  }

  if (particleSettings.inclusion) {
    std::cout << "Fraction: " << particleSettings.inclusion.value().fraction << std::endl;
    for (auto v : particleSettings.inclusion.value().ior) {
      std::cout << v.first << ": " << v.second.real() << " + " << v.second.imag() << std::endl;
    }
  }

  return 0;
}
