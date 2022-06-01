////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LIMB_DARKENING_HPP
#define LIMB_DARKENING_HPP

#include <cuda_runtime.h>

struct LimbDarkening {
  void __host__ __device__   init();
  double __host__ __device__ get(double r) const;

  double average = 1.0;
};

#endif // LIMB_DARKENING_HPP