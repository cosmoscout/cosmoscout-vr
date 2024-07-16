////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>

// This is used to pass the command line options to the Cuda kernel.
struct ShadowSettings {
  uint32_t size            = 512;
  bool     includeUmbra    = false;
  double   mappingExponent = 1.0;
};

#endif // TYPES_HPP