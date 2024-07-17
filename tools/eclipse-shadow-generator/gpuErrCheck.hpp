////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef GPU_ERR_CHECK_HPP
#define GPU_ERR_CHECK_HPP

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

// This macro is used in multiple locations to check for Cuda errors.
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans)                                                                             \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " (" << line << ")"
              << std::endl;
    if (abort) {
      exit(code);
    }
  }
}

#endif // GPU_ERR_CHECK_HPP