////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PARAMS_HPP
#define PARAMS_HPP

#include "../../../../src/cs-core/Settings.hpp"

/// The default values for the preprocessor parameters further down this file are based on the
/// parameters from Eric Bruneton:
/// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/constants.h
struct Params {

  /// This stores file paths to the CSV files containing the respective data. See the README of
  /// this preprocessor for a more detailed description.
  struct ScatteringComponent {
    std::string mPhaseFile;
    std::string mBetaScaFile;
    std::string mBetaAbsFile;
    std::string mDensityFile;

    /// The outer vector contains entries for each angle of the phase function. The first item
    /// corresponds to 0° (forward scattering), the last item to 180° (back scattering). The inner
    /// vectors contain the intensity values for each wavelength at the specific angle.
    std::vector<std::vector<float>> mPhase;

    /// Beta_sca per wavelength for the altitude where density is 1.0.
    std::vector<float> mScattering;

    /// Beta_abs per wavelength for the altitude where density is 1.0.
    std::vector<float> mAbsorption;

    /// Linear function describing the density distribution from bottom to top. The value at a
    /// specific altitude will be multiplied with the Beta_sca and Beta_abs values above.
    std::vector<float> mDensity;
  };

  /// This stores file paths to the CSV files containing the respective data. See the README of
  /// this preprocessor for a more detailed description.
  struct AbsorbingComponent {
    std::string mBetaAbsFile;
    std::string mDensityFile;

    /// Beta_abs per wavelength for N_0
    std::vector<float> mAbsorption;

    /// Linear function describing the density distribution from bottom to top. The value at a
    /// specific altitude will be multiplied with the Beta_sca and Beta_abs values above.
    std::vector<float> mDensity;
  };

  /// In this model, an atmosphere can consist out of three particle types. Two of them can
  /// scatter light, one can only absorb light. The former are usually used for small molecules
  /// and larger aerosols respectively, while the latter is used for ozone.
  ScatteringComponent               mMolecules;
  ScatteringComponent               mAerosols;
  std::optional<AbsorbingComponent> mOzone;

  /// To compute the refraction of light in the atmosphere, the refractive index of the atmosphere
  /// is needed. For increased precision, the refractive index is stored as n-1.
  float mRefractiveIndex = 0.0002777F;

  /// The wavelength values, in nanometers, and sorted in increasing order, for which the
  /// phase functions and extinction coefficients in the atmosphere components are given.
  std::vector<float> mWavelengths;

  float mSunDistance = 149600000000.F;
  float mMinAltitude = 6371000.F;
  float mMaxAltitude = 6471000.F;

  /// Refract the light when it travels through the atmosphere. This will produce an additional
  /// look-up texture in the same parameter space as the transmittance texture. For each sample,
  /// it contains the wavelength-dependent angular deviation of the light ray due to refraction.
  cs::utils::DefaultProperty<bool> mRefraction{true};

  /// The average reflectance of the ground used during multiple scattering.
  cs::utils::DefaultProperty<float> mGroundAlbedo{0.1F};

  /// The number of multiple scattering events to precompute. Use zero for single-scattering only.
  cs::utils::DefaultProperty<int32_t> mMultiScatteringOrder{4};

  /// The number of samples to evaluate when precomputing the optical depth. If refraction is used,
  /// the algorithm uses a fixed step size instead of a fixed number of samples. So if mRefraction
  /// is true, mStepSizeOpticalDepth (in meters) will be used instead of mSampleCountOpticalDepth.
  cs::utils::DefaultProperty<int32_t> mSampleCountOpticalDepth{500};
  cs::utils::DefaultProperty<int32_t> mStepSizeOpticalDepth{10000};

  /// The number of samples to evaluate when precomputing the single scattering. Larger values
  /// improve the sampling of thin atmospheric layers. If refraction is used, the algorithm uses a
  /// fixed step size instead of a fixed number of samples. So if mRefraction is true,
  /// mStepSizeSingleScattering (in meters) will be used instead of mSampleCountSingleScattering.
  cs::utils::DefaultProperty<int32_t> mSampleCountSingleScattering{50};
  cs::utils::DefaultProperty<int32_t> mStepSizeSingleScattering{10000};

  /// The number of samples to evaluate when precomputing the multiple scattering. Larger values
  /// tend to darken the horizon for thick atmospheres. If refraction is used, the algorithm uses a
  /// fixed step size instead of a fixed number of samples. So if mRefraction is true,
  /// mStepSizeMultiScattering (in meters) will be used instead of mSampleCountMultiScattering.
  cs::utils::DefaultProperty<int32_t> mSampleCountMultiScattering{50};
  cs::utils::DefaultProperty<int32_t> mStepSizeMultiScattering{10000};

  /// The number of samples to evaluate when precomputing the scattering density. Larger values
  /// spread out colors in the sky.
  cs::utils::DefaultProperty<int32_t> mSampleCountScatteringDensity{16};

  /// The number of samples to evaluate when precomputing the indirect irradiance.
  cs::utils::DefaultProperty<int32_t> mSampleCountIndirectIrradiance{32};

  /// The resolution of the transmittance texture. Larger values can improve the sampling of thin
  /// atmospheric layers close to the horizon.
  cs::utils::DefaultProperty<int32_t> mTransmittanceTextureWidth{256};
  cs::utils::DefaultProperty<int32_t> mTransmittanceTextureHeight{64};

  /// Larger values improve sampling of thick low-altitude layers.
  cs::utils::DefaultProperty<int32_t> mScatteringTextureRSize{32};

  /// Larger values reduce circular banding artifacts around zenith for thick atmospheres.
  cs::utils::DefaultProperty<int32_t> mScatteringTextureMuSize{128};

  /// Larger values reduce banding in the day-night transition when seen from space.
  cs::utils::DefaultProperty<int32_t> mScatteringTextureMuSSize{32};

  /// Larger values reduce circular banding artifacts around sun for thick atmospheres.
  cs::utils::DefaultProperty<int32_t> mScatteringTextureNuSize{8};

  /// The resolution of the irradiance texture.
  cs::utils::DefaultProperty<int32_t> mIrradianceTextureWidth{64};
  cs::utils::DefaultProperty<int32_t> mIrradianceTextureHeight{16};

  /// The maximum Sun zenith angle for which atmospheric scattering must be precomputed, in radians
  /// (for maximum precision, use the smallest Sun zenith angle yielding negligible sky light
  /// radiance values. For instance, for the Earth case, 102 degrees is a good choice for most cases
  /// (120 degrees is necessary for very high exposure values).
  cs::utils::DefaultProperty<float> mMaxSunZenithAngle{120.F / 180.F * glm::pi<float>()};
};

void from_json(nlohmann::json const& j, Params& o);

#endif // PARAMS_HPP
