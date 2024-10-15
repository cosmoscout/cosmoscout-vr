////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_MODELS_BRUNETON_METADATA_HPP
#define CSP_ATMOSPHERES_MODELS_BRUNETON_METADATA_HPP

#include "../../../../src/cs-core/Settings.hpp"

#include <glm/glm.hpp>

namespace csp::atmospheres::models::bruneton {

/// Besides the precomputed textures, the precomputation step also generates a metadata file which
/// contains some additional information required for rendering. This struct represents the content
/// of that file.
struct Metadata {
  /// The angular radius of the Sun in radians.
  float mSunAngularRadius{};

  /// The RGB illuminance of the Sun in lux.
  glm::vec3 mSunIlluminance{};

  /// As 4D textures are stored as layered 2D textures, we need this value to calculate the four
  /// dimensions of the 4D textures. This basically determines how many 2D textures are packed
  /// horizontally in each layer.
  int32_t mScatteringTextureNuSize{};

  /// The maximum Sun zenith angle for which atmospheric scattering is specified during the
  /// precomputation step and passed to the plugin here.
  float mMaxSunZenithAngle{};
};

void from_json(nlohmann::json const& j, Metadata& o);

} // namespace csp::atmospheres::models::bruneton

#endif // CSP_ATMOSPHERES_MODELS_BRUNETON_METADATA_HPP
