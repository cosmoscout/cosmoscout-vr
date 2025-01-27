////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: BSD-3-Clause

#include "Params.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Params::ScatteringComponent& o) {
  cs::core::Settings::deserialize(j, "betaSca", o.mBetaScaFile);
  cs::core::Settings::deserialize(j, "betaAbs", o.mBetaAbsFile);
  cs::core::Settings::deserialize(j, "phase", o.mPhaseFile);
  cs::core::Settings::deserialize(j, "density", o.mDensityFile);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Params::AbsorbingComponent& o) {
  cs::core::Settings::deserialize(j, "betaAbs", o.mBetaAbsFile);
  cs::core::Settings::deserialize(j, "density", o.mDensityFile);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Params& o) {
  cs::core::Settings::deserialize(j, "sunDistance", o.mSunDistance);
  cs::core::Settings::deserialize(j, "molecules", o.mMolecules);
  cs::core::Settings::deserialize(j, "aerosols", o.mAerosols);
  cs::core::Settings::deserialize(j, "ozone", o.mOzone);
  cs::core::Settings::deserialize(j, "ior", o.mRefractiveIndex);
  cs::core::Settings::deserialize(j, "minAltitude", o.mMinAltitude);
  cs::core::Settings::deserialize(j, "maxAltitude", o.mMaxAltitude);
  cs::core::Settings::deserialize(j, "refraction", o.mRefraction);
  cs::core::Settings::deserialize(j, "groundAlbedo", o.mGroundAlbedo);
  cs::core::Settings::deserialize(j, "multiScatteringOrder", o.mMultiScatteringOrder);
  cs::core::Settings::deserialize(j, "sampleCountOpticalDepth", o.mSampleCountOpticalDepth);
  cs::core::Settings::deserialize(j, "stepSizeOpticalDepth", o.mStepSizeOpticalDepth);
  cs::core::Settings::deserialize(j, "sampleCountSingleScattering", o.mSampleCountSingleScattering);
  cs::core::Settings::deserialize(j, "stepSizeSingleScattering", o.mStepSizeSingleScattering);
  cs::core::Settings::deserialize(j, "sampleCountMultiScattering", o.mSampleCountMultiScattering);
  cs::core::Settings::deserialize(j, "stepSizeMultiScattering", o.mStepSizeMultiScattering);
  cs::core::Settings::deserialize(
      j, "sampleCountScatteringDensity", o.mSampleCountScatteringDensity);
  cs::core::Settings::deserialize(
      j, "sampleCountIndirectIrradiance", o.mSampleCountIndirectIrradiance);
  cs::core::Settings::deserialize(j, "transmittanceTextureWidth", o.mTransmittanceTextureWidth);
  cs::core::Settings::deserialize(j, "transmittanceTextureHeight", o.mTransmittanceTextureHeight);
  cs::core::Settings::deserialize(j, "scatteringTextureRSize", o.mScatteringTextureRSize);
  cs::core::Settings::deserialize(j, "scatteringTextureMuSize", o.mScatteringTextureMuSize);
  cs::core::Settings::deserialize(j, "scatteringTextureMuSSize", o.mScatteringTextureMuSSize);
  cs::core::Settings::deserialize(j, "scatteringTextureNuSize", o.mScatteringTextureNuSize);
  cs::core::Settings::deserialize(j, "irradianceTextureWidth", o.mIrradianceTextureWidth);
  cs::core::Settings::deserialize(j, "irradianceTextureHeight", o.mIrradianceTextureHeight);
  cs::core::Settings::deserialize(j, "maxSunZenithAngle", o.mMaxSunZenithAngle);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
