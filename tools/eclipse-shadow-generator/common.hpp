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

struct MappingSettings {
  bool   mIncludeUmbra = false;
  double mExponent     = 1.0;
};

struct OutputSettings {
  std::string mFile = "shadow.hdr";
  uint32_t    mSize = 512;
  float*      mBuffer;
};

struct GeometrySettings {
  double mRadiusOcc  = 6370900.0;
  double mRadiusAtmo = 6451000.0;
  double mRadiusSun  = 696340000.0;
  double mSunOccDist = 149597870700.0;
};

void addMappingSettingsFlags(cs::utils::CommandLine& commandLine, MappingSettings& settings);
void addOutputSettingsFlags(cs::utils::CommandLine& commandLine, OutputSettings& settings);
void addGeometrySettingsFlags(cs::utils::CommandLine& commandLine, GeometrySettings& settings);

} // namespace common

#endif // COMMON_HPP