////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_CSVLOADER_HPP
#define CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_CSVLOADER_HPP

#include <functional>
#include <string>
#include <vector>

namespace csp::atmospheres::models::bruneton::internal {

/// This static class is used to load the phase functions, scattering & absorption coefficients, as
/// well as the density distributions of atmospheric particles from CSV files.
class CSVLoader {
 public:
  /// This loads a density distribution sampled at evenly spaced altitudes from the given CSV file.
  /// The first returned value corresponds to the bottom-most density in the atmosphere. If the
  /// densityCount parameter is initialized to 0, it will be filled with the number of densities
  /// read from the file. If it is initialized to a value != 0, the method will compare the number
  /// of read densities to this number and throw an exception if this doesn't match. This is used to
  /// ensure that all constituents of an atmosphere are sampled at the same altitudes. The content
  /// of the CSV file could look like this:
  ///
  /// density
  /// 1.0
  /// 0.9369102299935942
  /// 0.8778007790666494
  /// ...
  /// 0.0
  ///
  static std::vector<double> readDensity(std::string const& filename, uint32_t& densityCount);

  /// This loads a 2D wavelength-dependent phase function sampled a specific angles and wavelengths
  /// from a CSV file. The method basically returns the columns from the CSV file. This means, for
  /// each phase function angle, the outer vector contains a spectrum of intensity values. If the
  /// wavelengths parameter is initially empty, it will be filled with the wavelengths read from the
  /// file. If it already contains some values, the method will compare them to the newly read
  /// wavelengths and throw an exception if they don't match. This is used to ensure that all
  /// constituents of an atmosphere are sampled at the same wavelengths. The units for the
  /// wavelengths should be [m]. The content of the CSV file could look like this:
  ///
  /// lambda,  0.0,   1.0,   2.0,   ..., 180.0
  /// 4.4e-07, 0.933, 0.931, 0.923, ..., 0.022
  /// 5.5e-07, 0.257, 0.257, 0.256, ..., 0.025
  /// 6.8e-07, 0.091, 0.091, 0.091, ..., 0.045
  ///
  static std::vector<std::vector<double>> readPhase(
      std::string const& filename, std::vector<double>& wavelengths);

  /// This loads wavelength-dependent scattering coefficients (beta_sca) or absorption coefficients
  /// (beta_abs) from CSV files. If the wavelengths parameter is initially empty, it will be filled
  /// with the wavelengths read from the file. If it already contains some values, the method will
  /// compare them to the newly read wavelengths and throw an exception if they don't match. This is
  /// used to ensure that all constituents of an atmosphere are sampled at the same wavelengths. The
  /// units for the wavelengths should be [m], for the extinction coefficients [m^-1]. The content
  /// of the CSV file could look like this:
  ///
  /// lambda,  beta_sca
  /// 4.4e-07, 5.70e-07
  /// 5.5e-07, 2.33e-07
  /// 6.8e-07, 1.00e-07
  ///
  static std::vector<double> readExtinction(
      std::string const& filename, std::vector<double>& wavelengths);
};

} // namespace csp::atmospheres::models::bruneton::internal

#endif // CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_CSVLOADER_HPP