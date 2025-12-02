////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Metadata.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::atmospheres::models::bruneton {

void from_json(nlohmann::json const& j, Metadata& o) {
  cs::core::Settings::deserialize(j, "sunAngularRadius", o.mSunAngularRadius);
  cs::core::Settings::deserialize(j, "sunIlluminance", o.mSunIlluminance);
  cs::core::Settings::deserialize(j, "scatteringTextureNuSize", o.mScatteringTextureNuSize);
  cs::core::Settings::deserialize(j, "maxSunZenithAngle", o.mMaxSunZenithAngle);
  cs::core::Settings::deserialize(j, "refraction", o.mRefraction);
}

} // namespace csp::atmospheres::models::bruneton

////////////////////////////////////////////////////////////////////////////////////////////////////
