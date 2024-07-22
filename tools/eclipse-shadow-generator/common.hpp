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

struct ShadowSettings {
  bool   includeUmbra    = false;
  double mappingExponent = 1.0;
};

struct OutputSettings {
  std::string output = "shadow.hdr";
  uint32_t    size   = 512;
};

void addShadowSettingsFlags(cs::utils::CommandLine& commandLine, ShadowSettings& settings);

void addOutputSettingsFlags(cs::utils::CommandLine& commandLine, OutputSettings& settings);

} // namespace common

#endif // COMMON_HPP