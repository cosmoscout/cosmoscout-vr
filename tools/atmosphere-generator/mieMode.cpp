////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../../src/cs-utils/CommandLine.hpp"
#include "bhmie.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>

#include <complex>
#include <fstream>
#include <map>
#include <random>

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
  double  minLambda     = 0.36;
  double  maxLambda     = 0.83;
  int32_t lambdaSteps   = 15;
  int32_t thetaSteps    = 1024;
  int32_t radiusSamples = 1000;
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

struct MieResult {
  std::vector<double> phase;
  double              cSca;
  double              cAbs;
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

std::complex<double> getRefractiveIndex(double lambda, IoRSpectrum const& iorSpectrum) {

  if (lambda <= iorSpectrum.begin()->first) {
    return iorSpectrum.begin()->second;
  }

  if (lambda >= iorSpectrum.rbegin()->first) {
    return iorSpectrum.rbegin()->second;
  }

  auto upper = iorSpectrum.lower_bound(lambda);
  auto lower = upper;
  if (lower != iorSpectrum.begin()) {
    --lower;
  }

  double alpha = (lambda - lower->first) / (upper->first - lower->first);
  return lower->second + alpha * (upper->second - lower->second);
}

std::complex<double> maxwellGarnett(
    std::complex<double> inclusionIoR, double substrateIoR, double fraction) {
  auto inclusionIoR2 = inclusionIoR * inclusionIoR;
  auto substrateIoR2 = substrateIoR * substrateIoR;

  auto n2 =
      substrateIoR2 *
      (inclusionIoR2 + 2.f * substrateIoR2 + 2.f * fraction * (inclusionIoR2 - substrateIoR2)) /
      (inclusionIoR2 + 2.f * substrateIoR2 - fraction * (inclusionIoR2 - substrateIoR2));

  return std::sqrt(n2);
}

std::vector<double> fillVector(double min, double max, size_t count) {
  std::vector<double> result(count);

  if (count == 1) {
    result[0] = min;
    return result;
  }

  for (size_t i(0); i < count; ++i) {
    result[i] = min + (max - min) * i / (count - 1.0);
  }

  return result;
}

MieResult mieDisperse(
    int32_t thetaSteps, double lambda, std::complex<double> ior, std::vector<double> const& radii) {

  MieResult result;
  result.phase = std::vector<double>(2 * thetaSteps - 1);
  result.cAbs  = 0.0;
  result.cSca  = 0.0;

  std::vector<std::complex<double>> cxs1(2 * thetaSteps);
  std::vector<std::complex<double>> cxs2(2 * thetaSteps);

  double totalWeight = 0.0;

  for (double r : radii) {
    double x = 2.0 * r * glm::pi<double>() / lambda;
    double qext, qsca, qback, gsca;

    bhmie(x, ior, thetaSteps, cxs1, cxs2, &qext, &qsca, &qback, &gsca);

    double normalization = glm::pi<double>() * x * x * qsca;

    double csca = qsca * glm::pi<double>() * r * r;
    double cext = qext * glm::pi<double>() * r * r;
    totalWeight += csca;

    for (int32_t i(0); i < thetaSteps * 2 - 1; ++i) {
      double intensity =
          0.5 * (std::pow(std::abs(cxs1[i + 1]), 2.f) + std::pow(std::abs(cxs2[i + 1]), 2.f));

      result.phase[i] += intensity / normalization * csca;
    }

    result.cSca += csca;
    result.cAbs += cext - csca;
  }

  for (auto& p : result.phase) {
    p /= totalWeight;
  }

  result.cSca /= radii.size();
  result.cAbs /= radii.size();

  return result;
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
      "The minimum wavelength in µm (default: " + std::to_string(samplingSettings.minLambda) +
          ").");
  args.addArgument({"--max-lambda"}, &samplingSettings.maxLambda,
      "The maximum wavelength in µm (default: " + std::to_string(samplingSettings.maxLambda) +
          ").");
  args.addArgument({"--lambda-steps"}, &samplingSettings.lambdaSteps,
      "The number of wavelengths to compute (default: " +
          std::to_string(samplingSettings.lambdaSteps) + ").");
  args.addArgument({"--theta-steps"}, &samplingSettings.thetaSteps,
      "The number of angles to compute (default: " + std::to_string(samplingSettings.thetaSteps) +
          ").");
  args.addArgument({"--radius-samples"}, &samplingSettings.radiusSamples,
      "The number of particles to compute per size mode (default: " +
          std::to_string(samplingSettings.radiusSamples) + ").");
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

  std::vector<double> lambdas = fillVector(
      samplingSettings.minLambda, samplingSettings.maxLambda, samplingSettings.lambdaSteps);

  std::vector<std::complex<double>> ior(samplingSettings.lambdaSteps);

  for (int32_t i(0); i < samplingSettings.lambdaSteps; ++i) {
    double lambda = lambdas[i];
    if (particleSettings.inclusion) {
      auto   inclusionIoR = getRefractiveIndex(lambda, particleSettings.inclusion.value().ior);
      auto   substrateIoR = getRefractiveIndex(lambda, particleSettings.ior);
      double fraction     = particleSettings.inclusion.value().fraction;
      ior[i]              = maxwellGarnett(inclusionIoR, substrateIoR.real(), fraction);
    } else {
      ior[i] = getRefractiveIndex(lambda, particleSettings.ior);
    }
  }

  double totalModeWeight = 0.0;

  for (int32_t i(0); i < samplingSettings.lambdaSteps; ++i) {
    double              lambda = lambdas[i];
    std::vector<double> phase(samplingSettings.thetaSteps * 2 - 1);

    for (auto sizeMode : particleSettings.sizeModes) {
      std::random_device rd{};
      std::mt19937       gen{rd()};

      std::vector<double> radii(samplingSettings.radiusSamples);

      if (sizeMode.sizeDistribution == DistributionType::eGamma) {
        double shape = (1.0 - 2.0 * sizeMode.effectiveVariance) / sizeMode.effectiveVariance;
        double theta = sizeMode.effectiveRadius * sizeMode.effectiveVariance;
        std::gamma_distribution<> d(shape, theta);

        for (int32_t i(0); i < samplingSettings.radiusSamples; ++i) {
          radii[i] = d(gen);
        }

      } else if (sizeMode.sizeDistribution == DistributionType::eLogNormal) {

        double                        s2 = std::log(sizeMode.effectiveVariance + 1);
        double                        u  = std::log(sizeMode.effectiveRadius) - 2.5 * s2;
        std::lognormal_distribution<> d(u, std::sqrt(s2));

        for (int32_t i(0); i < samplingSettings.radiusSamples; ++i) {
          radii[i] = d(gen);
        }
      }

      auto   mieResult = mieDisperse(samplingSettings.thetaSteps, lambda, ior[i], radii);
      double weight    = sizeMode.relativeAmount * mieResult.cSca;
      totalModeWeight += weight;

      for (size_t i(0); i < mieResult.phase.size(); ++i) {
        phase[i] += mieResult.phase[i] * weight;
      }
    }

    std::cout << std::endl;
    std::cout << "Phase Function for lambda = " << lambda << " µm" << std::endl;
    for (double p : phase) {
      std::cout << p / totalModeWeight << std::endl;
    }
  }

  return 0;
}
