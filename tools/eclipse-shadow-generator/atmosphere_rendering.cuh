////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef ATMOSPHERE_RENDERING_HPP
#define ATMOSPHERE_RENDERING_HPP

#include "common.hpp"
#include "gpuErrCheck.hpp"
#include "math.cuh"

namespace advanced {

// These input textures are required for the Bruneton precomputed atmospheric scattering model.
// They have to be precomputed using the "bruneton-preprocessor" tool of the csp-atmospheres plugin.
struct Textures {
  cudaTextureObject_t mPhase;
  cudaTextureObject_t mThetaDeviation;
  cudaTextureObject_t mTransmittance;
  cudaTextureObject_t mMultipleScattering;
  cudaTextureObject_t mSingleAerosolsScattering;

  int    mTransmittanceTextureWidth;
  int    mTransmittanceTextureHeight;
  int    mScatteringTextureMuSize;
  int    mScatteringTextureMuSSize;
  int    mScatteringTextureNuSize;
  double mMuSMin;
};

// Loads all required textures from the given output directory from the Bruneton preprocessor tool.
Textures loadTextures(std::string const& path);

// Computes the luminance of the atmosphere for the given geometry.
__device__ glm::vec3 getLuminance(glm::dvec3 camera, glm::dvec3 viewRay, glm::dvec3 sunDirection,
    common::Geometry const& geometry, common::LimbDarkening const& limbDarkening,
    Textures const& textures, double phiSun);

} // namespace advanced

#endif // ATMOSPHERE_RENDERING_HPP