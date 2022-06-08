////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LIMB_DARKENING_HPP
#define LIMB_DARKENING_HPP

#include <cuda_runtime.h>

/// This struct implements a simple wavelength-independent limb darkening model.
class LimbDarkening {

 public:
  /// This computes the average brightness over the entire solar disc by sampling so that the get()
  /// method can return normalized values.
  void __host__ __device__ init();

  /// Returns the Sun's brightness at the given radial distance to the center of the solar disc
  /// between [0...1]. The returned values are normalized so that the average brightness over entire
  /// disc is one.
  double __host__ __device__ get(double r) const;

 private:
  double mAverage = 1.0;
};

#endif // LIMB_DARKENING_HPP