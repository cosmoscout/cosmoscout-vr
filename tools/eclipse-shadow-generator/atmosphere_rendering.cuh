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

__host__ Textures loadTextures(std::string const& path);

__device__ glm::vec3 getLuminance(glm::dvec3 camera, glm::dvec3 viewRay, glm::dvec3 sunDirection,
    common::Geometry const& geometry, common::LimbDarkening const& limbDarkening,
    Textures const& textures, double phiSun);

} // namespace advanced

#endif // ATMOSPHERE_RENDERING_HPP