////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef COMMON_HPP
#define COMMON_HPP

#include "../../../src/cs-utils/CommandLine.hpp"

#include <cstdint>

namespace common {

// This is used to describe the different shadow-map parameterization variants described in figure
// 10 of "Real-Time Rendering of Eclipses without Incorporation of Atmospheric Effects". The values
// can be set via command line arguments.
struct Mapping {
  bool   mIncludeUmbra = false;
  double mExponent     = 1.0;
};

// This is used to describe the output file and resolution. The values can be set via command line
// arguments. The buffer is allocated and filled by the shadow-map generator.
struct Output {
  std::string mFile = "shadow.hdr";
  uint32_t    mSize = 512;
  float*      mBuffer;
};

// When computing an eclipse shadow involving atmospheric effects, the geometry of the Sun,
// occluding body, and its atmosphere are needed. The values can be set via command line arguments.
struct Geometry {
  double mRadiusOcc  = 6370900.0;
  double mRadiusAtmo = 6451000.0;
  double mRadiusSun  = 696340000.0;
  double mSunOccDist = 149597870700.0;
};

// This adds the command line arguments for the shadow-map parameterization to the given
// CommandLine object.
void addMappingFlags(cs::utils::CommandLine& commandLine, Mapping& settings);

// This adds the command line arguments for the output file and resolution to the given
// CommandLine object.
void addOutputFlags(cs::utils::CommandLine& commandLine, Output& settings);

// This adds the command line arguments for the geometry of the Sun, occluding body, and its
// atmosphere to the given CommandLine object.
void addGeometryFlags(cs::utils::CommandLine& commandLine, Geometry& settings);

} // namespace common

#endif // COMMON_HPP