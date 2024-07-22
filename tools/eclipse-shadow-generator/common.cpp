////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "common.hpp"

namespace common {

void addShadowSettingsFlags(cs::utils::CommandLine& commandLine, ShadowSettings& settings) {
  commandLine.addArgument({"--with-umbra"}, &settings.includeUmbra,
      "Add the umbra region to the shadow map (default: " + std::to_string(settings.includeUmbra) +
          ").");
  commandLine.addArgument({"--mapping-exponent"}, &settings.mappingExponent,
      "Adjusts the distribution of sampling positions. A value of 1.0 will position the "
      "umbra's end in the middle of the texture, larger values will shift this to the "
      "right. (default: " +
          std::to_string(settings.mappingExponent) + ").");
}

void addOutputSettingsFlags(cs::utils::CommandLine& commandLine, OutputSettings& settings) {
  commandLine.addArgument({"-o", "--output"}, &settings.output,
      "The image will be written to this file (default: \"" + settings.output + "\").");
  commandLine.addArgument({"-s", "--size"}, &settings.size,
      "The output texture size (default: " + std::to_string(settings.size) + ").");
}

} // namespace common