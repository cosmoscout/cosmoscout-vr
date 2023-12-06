////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "mieMode.hpp"

#include "../../src/cs-utils/CommandLine.hpp"
#include "../../src/cs-utils/utils.hpp"
#include "bhmie.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>

#include <complex>
#include <fstream>
#include <map>
#include <random>

// Allow deserialization of std::complex types using nlohmann::json.
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

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////
// The particle sizes can be distributed according to various random distributions. For now,      //
// Gamma- and LogNormal distributions are supported. The modified variants take effective radii   //
// and variances as input parameters.                                                             //
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class DistributionType { eGamma, eModifiedGamma, eLogNormal, eModifiedLogNormal };

// Make the DistributionTypes available for JSON deserialization.
NLOHMANN_JSON_SERIALIZE_ENUM(
    DistributionType, {
                          {DistributionType::eGamma, "gamma"},
                          {DistributionType::eModifiedGamma, "modifiedGamma"},
                          {DistributionType::eLogNormal, "logNormal"},
                          {DistributionType::eModifiedLogNormal, "modifiedLogNormal"},
                      })

////////////////////////////////////////////////////////////////////////////////////////////////////
// This type describes a particle size distribution. Depending on the type, the parameters A and  //
// B will store different things.                                                                 //
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Distribution {
  DistributionType type;

  // Shape for gamma distribution, Mean for log-normal distribution, and Effective Radius for the
  // modified variants.
  double paramA;

  // Scale for gamma distribution, Sigma for log-normal distribution, and Effective Variance for the
  // modified variants.
  double paramB;

  // This is used in multi-modal distributions to compute the weight of this mode.
  double relativeNumberDensity;
};

// Make this type available for JSON deserialization.
void from_json(nlohmann::json const& j, Distribution& s) {
  j.at("type").get_to(s.type);
  j.at("relativeNumberDensity").get_to(s.relativeNumberDensity);

  switch (s.type) {
  case DistributionType::eGamma:
    j.at("shape").get_to(s.paramA);
    j.at("scale").get_to(s.paramB);
    break;
  case DistributionType::eModifiedGamma:
    j.at("effectiveRadius").get_to(s.paramA);
    j.at("effectiveVariance").get_to(s.paramB);
    break;
  case DistributionType::eLogNormal:
    j.at("mean").get_to(s.paramA);
    j.at("sigma").get_to(s.paramB);
    break;
  case DistributionType::eModifiedLogNormal:
    j.at("effectiveRadius").get_to(s.paramA);
    j.at("effectiveVariance").get_to(s.paramB);
    break;
  }
}

std::vector<double> sampleRadii(
    DistributionType type, int32_t count, double paramA, double paramB) {

  std::random_device rd{};
  std::mt19937       gen{rd()};

  std::vector<double> radii(count);

  if (type == DistributionType::eGamma || type == DistributionType::eModifiedGamma) {

    double shape = paramA;
    double scale = paramB;

    if (type == DistributionType::eModifiedGamma) {
      shape = (1.0 - 2.0 * paramB) / paramB;
      scale = paramA * paramB;
    }

    std::gamma_distribution<> d(shape, scale);

    for (int32_t i(0); i < count; ++i) {
      radii[i] = d(gen);
    }

  } else if (type == DistributionType::eLogNormal || type == DistributionType::eModifiedLogNormal) {

    double mean  = paramA;
    double sigma = paramB;

    if (type == DistributionType::eModifiedLogNormal) {
      double s2 = std::log(paramB + 1);
      mean      = std::log(paramA) - 2.5 * s2;
      sigma     = std::sqrt(s2);
    }

    std::lognormal_distribution<> d(mean, sigma);

    for (int32_t i(0); i < count; ++i) {
      radii[i] = d(gen);
    }
  }

  return radii;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// This type is used to store a wavelength-dependent complex refractive index. The key in the map //
// is a wavelength in µm.                                                                         //
////////////////////////////////////////////////////////////////////////////////////////////////////

typedef std::map<double, std::complex<double>> IoRSpectrum;

// This retrieves an index of refraction for a given lambda in µm from an IoRSpectrum by linear
// interpolation. The given lambda will be clamped to the covered spectrum.
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

////////////////////////////////////////////////////////////////////////////////////////////////////
// This struct describes a type of particles. The particle type is defined by a multi-modal size  //
// distribution and a potentially complex and wavelength-dependent index of refraction.           //
////////////////////////////////////////////////////////////////////////////////////////////////////

struct ParticleSettings {

  // The particle sizes can follow a mixture of various random distributions.
  std::vector<Distribution> sizeModes;

  // The potentially complex and wavelength-dependent index of refraction of the particles. If an
  // inclusion is given, the imaginary part of this index of refraction will be ignored.
  IoRSpectrum ior;

  // The particles can have a fraction of another material included. This will be incorporated into
  // the final IoR using the Maxwell-Garnett Mixing-Rule.
  struct Inclusion {
    double      fraction;
    IoRSpectrum ior;
  };

  std::optional<Inclusion> inclusion;
};

// Make this type available for JSON deserialization.
void from_json(nlohmann::json const& j, ParticleSettings::Inclusion& s) {
  j.at("fraction").get_to(s.fraction);
  j.at("ior").get_to(s.ior);
}

void from_json(nlohmann::json const& j, ParticleSettings& s) {
  j.at("sizeModes").get_to(s.sizeModes);
  j.at("ior").get_to(s.ior);

  if (j.contains("inclusion")) {
    ParticleSettings::Inclusion inclusion;
    j.at("inclusion").get_to(inclusion);
    s.inclusion = inclusion;
  }
}

// Computes the combined index of refraction of two materials. The inclusion material can have a
// complex index of refraction.
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

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

struct MieResult {
  std::vector<double> phase;
  double              cSca;
  double              cAbs;
};

MieResult mieDisperse(int32_t thetaSamples, double lambda, std::complex<double> ior,
    std::vector<double> const& radii) {

  MieResult result;
  result.phase = std::vector<double>(2 * thetaSamples - 1);
  result.cAbs  = 0.0;
  result.cSca  = 0.0;

  std::vector<std::complex<double>> cxs1(2 * thetaSamples);
  std::vector<std::complex<double>> cxs2(2 * thetaSamples);

  double totalWeight = 0.0;

  for (double r : radii) {
    double x = 2.0 * r * glm::pi<double>() / lambda;
    double qext, qsca, qback, gsca;

    bhmie(x, ior, thetaSamples, cxs1, cxs2, &qext, &qsca, &qback, &gsca);

    double normalization = glm::pi<double>() * x * x * qsca;

    double csca = qsca * glm::pi<double>() * r * r;
    double cext = qext * glm::pi<double>() * r * r;
    totalWeight += csca;

    for (int32_t i(0); i < thetaSamples * 2 - 1; ++i) {
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

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

int mieMode(std::vector<std::string> const& arguments) {

  bool        cPrintHelp     = false;
  std::string cInput         = "";
  std::string cOutput        = "particles.csv";
  std::string cLambdas       = "";
  double      cMinLambda     = 0.36;
  double      cMaxLambda     = 0.83;
  int32_t     cLambdaSamples = 15;
  int32_t     cThetaSamples  = 91;
  int32_t     cRadiusSamples = 1000;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Welcome to the Mie preprocessor! Here are the available options:");
  args.addArgument(
      {"-i", "--input"}, &cInput, "The JSON file with the particle information (required).");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The scattering data will be written to this CSV file (default: \"" + cOutput + "\").");
  args.addArgument({"--min-lambda"}, &cMinLambda,
      "The minimum wavelength in µm (default: " + std::to_string(cMinLambda) + ").");
  args.addArgument({"--max-lambda"}, &cMaxLambda,
      "The maximum wavelength in µm (default: " + std::to_string(cMaxLambda) + ").");
  args.addArgument({"--lambda-samples"}, &cLambdaSamples,
      "The number of wavelengths to compute (default: " + std::to_string(cLambdaSamples) + ").");
  args.addArgument({"--lambdas"}, &cLambdas,
      "A comma-separated list of wavelengths in µm. If provided, --min-lambda, --max-lambda, and "
      "--lambda-samples are ignored.");
  args.addArgument({"--theta-samples"}, &cThetaSamples,
      "The number of angles to compute between 0° and 90° (default: " +
          std::to_string(cThetaSamples) + ").");
  args.addArgument({"--radius-samples"}, &cRadiusSamples,
      "The number of particles to compute per size mode (default: " +
          std::to_string(cRadiusSamples) + ").");
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

  std::vector<double> lambdas;

  if (cLambdas.empty()) {
    if (cLambdaSamples <= 0) {
      std::cerr << "Lmabda samples must be > 0!" << std::endl;
      return 1;
    } else if (cLambdaSamples == 1) {
      lambdas.push_back(cMinLambda);
    } else {
      for (int32_t i(0); i < cLambdaSamples; ++i) {
        lambdas.push_back(cMinLambda + (cMaxLambda - cMinLambda) * i / (cLambdaSamples - 1.0));
      }
    }

  } else {
    auto tokens = cs::utils::splitString(cLambdas, ',');
    for (auto token : tokens) {
      lambdas.push_back(cs::utils::fromString<double>(token));
    }
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

  std::vector<std::complex<double>> ior(lambdas.size());

  for (size_t i(0); i < lambdas.size(); ++i) {
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

  std::ofstream output(cOutput);

  int32_t totalAngles = cThetaSamples * 2 - 1;

  output << "lambda,c_sca,c_abs";

  for (int32_t t(0); t < totalAngles; ++t) {
    output << "," << 180.0 * t / (totalAngles - 1.0);
  }

  output << std::endl;

  for (size_t l(0); l < lambdas.size(); ++l) {

    double lambda = lambdas[l];

    std::vector<double> phase(totalAngles);
    double              cSca = 0.0;
    double              cAbs = 0.0;

    double totalCoeffWeight = 0.0;
    double totalPhaseWeight = 0.0;

    for (auto sizeMode : particleSettings.sizeModes) {
      auto radii = sampleRadii(sizeMode.type, cRadiusSamples, sizeMode.paramA, sizeMode.paramB);

      auto mieResult = mieDisperse(cThetaSamples, lambda, ior[l], radii);

      double coeffWeight = sizeMode.relativeNumberDensity;
      double phaseWeight = coeffWeight * mieResult.cSca;

      totalCoeffWeight += coeffWeight;
      totalPhaseWeight += phaseWeight;

      cSca += mieResult.cSca * coeffWeight;
      cAbs += mieResult.cAbs * coeffWeight;

      for (size_t i(0); i < mieResult.phase.size(); ++i) {
        phase[i] += mieResult.phase[i] * phaseWeight;
      }
    }

    output << lambda << "," << cSca / totalCoeffWeight << "," << cAbs / totalCoeffWeight;

    for (double p : phase) {
      output << "," << p / totalPhaseWeight;
    }

    output << std::endl;
  }

  return 0;
}
