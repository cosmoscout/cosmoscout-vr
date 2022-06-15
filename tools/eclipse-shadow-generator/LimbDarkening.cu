////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LimbDarkening.cuh"
#include "math.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////

void __host__ __device__ LimbDarkening::init() {
  double        totalBrightness = 0.0;
  const int32_t samples         = 2048;

  glm::dvec2 samplePos;

  // Compute the average brightness by sampling over the entire disc. Thanks to the symmetry of the
  // disc, we only need to sample a quadrant.
  for (int32_t y = 0; y < samples; ++y) {
    samplePos.y = (1.0 * y + 0.5) / samples;

    for (int32_t x(0); x < samples; ++x) {
      samplePos.x = (1.0 * x + 0.5) / samples;
      totalBrightness += get(glm::length(samplePos));
    }
  }

  mAverage = totalBrightness / glm::pi<double>() / std::pow(samples / 2, 2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double __host__ __device__ LimbDarkening::get(double r) const {
  return r >= 1.0 ? 0.0 : (1.0 - 0.6 * (1.0 - std::sqrt(1 - r * r))) / mAverage;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
