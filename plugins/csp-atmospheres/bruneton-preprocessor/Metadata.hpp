////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef METADATA_HPP
#define METADATA_HPP

#include "../../../../src/cs-core/Settings.hpp"

#include <glm/glm.hpp>

/// The preprocessor not only generates textures, but also some metadata which is needed to render
/// the atmosphere. This struct is used to store this metadata.
struct Metadata {
  /// The angular radius of the Sun as well as the RGB illuminance at the planet's average distance
  /// to the Sun is required during rendering.
  float     mSunAngularRadius{};
  glm::vec3 mSunIlluminance{};

  /// As the scattering textures are 4D textures stored in 3D textures, we need to know the number
  /// of textures packed next to each other in the layers of the 3D texture.
  int32_t mScatteringTextureNuSize{};

  /// The maximum Sun zenith angle for which atmospheric scattering was be precomputed.
  float mMaxSunZenithAngle{};

  /// Whether refraction was used during preprocessing.
  bool mRefraction{};
};

void to_json(nlohmann::json& j, Metadata const& o);

#endif // METADATA_HPP
