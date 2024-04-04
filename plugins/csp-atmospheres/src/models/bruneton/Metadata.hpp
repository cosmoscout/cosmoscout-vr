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

struct Metadata {
  /// The angular radius of the Sun needs to be specified. As SPICE is not fully available when
  /// the plugin is loaded, we cannot compute it. Also, this actually varies in reality.
  float     mSunAngularRadius{};
  glm::vec3 mSunIlluminance{};

  /// Larger values reduce circular banding artifacts around sun for thick atmospheres.
  int32_t mScatteringTextureNuSize{};

  /// The maximum Sun zenith angle for which atmospheric scattering must be precomputed, in
  /// radians (for maximum precision, use the smallest Sun zenith angle yielding negligible sky
  /// light radiance values. For instance, for the Earth case, 102 degrees is a good choice for
  /// most cases (120 degrees is necessary for very high exposure values).
  float mMaxSunZenithAngle{};
};

void from_json(nlohmann::json const& j, Metadata& o);

} // namespace csp::atmospheres::models::bruneton

#endif // CSP_ATMOSPHERES_MODELS_BRUNETON_METADATA_HPP
