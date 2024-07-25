////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef COMMON_HPP
#define COMMON_HPP

#include "../../../src/cs-utils/CommandLine.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Common functionality which is used by multiple modes.                                          //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstdint>

namespace common {

struct Mapping {
  bool   mIncludeUmbra = false;
  double mExponent     = 1.0;
};

struct Output {
  std::string mFile = "shadow.hdr";
  uint32_t    mSize = 512;
  float*      mBuffer;
};

struct Geometry {
  double mRadiusOcc  = 6370900.0;
  double mRadiusAtmo = 6451000.0;
  double mRadiusSun  = 696340000.0;
  double mSunOccDist = 149597870700.0;
};

void addMappingFlags(cs::utils::CommandLine& commandLine, Mapping& settings);
void addOutputFlags(cs::utils::CommandLine& commandLine, Output& settings);
void addGeometryFlags(cs::utils::CommandLine& commandLine, Geometry& settings);

} // namespace common

#endif // COMMON_HPP