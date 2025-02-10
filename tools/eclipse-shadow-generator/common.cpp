////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "common.hpp"

namespace common {

////////////////////////////////////////////////////////////////////////////////////////////////////

void addMappingFlags(cs::utils::CommandLine& commandLine, Mapping& settings) {
  commandLine.addArgument({"--with-umbra"}, &settings.mIncludeUmbra,
      "Add the umbra region to the shadow map (default: " + std::to_string(settings.mIncludeUmbra) +
          ").");
  commandLine.addArgument({"--mapping-exponent"}, &settings.mExponent,
      "Adjusts the distribution of sampling positions. A value of 1.0 will position the "
      "umbra's end in the middle of the texture, larger values will shift this to the "
      "right. (default: " +
          std::to_string(settings.mExponent) + ").");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void addOutputFlags(cs::utils::CommandLine& commandLine, Output& settings) {
  commandLine.addArgument({"-o", "--output"}, &settings.mFile,
      "The image will be written to this file (default: \"" + settings.mFile + "\").");
  commandLine.addArgument({"-s", "--size"}, &settings.mSize,
      "The output texture size (default: " + std::to_string(settings.mSize) + ").");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void addGeometryFlags(cs::utils::CommandLine& commandLine, Geometry& settings) {
  commandLine.addArgument({"--radius-occ"}, &settings.mRadiusOcc,
      "The radius of the occulting body in meters (default: " +
          std::to_string(settings.mRadiusOcc) + ").");
  commandLine.addArgument({"--radius-atmo"}, &settings.mRadiusAtmo,
      "The radius of the atmosphere in meters (default: " + std::to_string(settings.mRadiusAtmo) +
          ").");
  commandLine.addArgument({"--radius-sun"}, &settings.mRadiusSun,
      "The radius of the Sun in meters (default: " + std::to_string(settings.mRadiusSun) + ").");
  commandLine.addArgument({"--sun-occ-dist"}, &settings.mSunOccDist,
      "The distance between the sun and the occluding body in meters (default: " +
          std::to_string(settings.mSunOccDist) + ").");
  commandLine.addArgument({"--average-terrain-height"}, &settings.mAverageTerrainHeight,
      "The average height of opaque geometry blocking light, in meters (default: " +
          std::to_string(settings.mAverageTerrainHeight) + ").");
  commandLine.addArgument({"--cloud-altitude"}, &settings.mCloudAltitude,
      "No light will pass below this altitude, in meters (default: " +
          std::to_string(settings.mCloudAltitude) + ").");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace common