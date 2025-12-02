////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Metadata.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

void to_json(nlohmann::json& j, Metadata const& o) {
  cs::core::Settings::serialize(j, "sunAngularRadius", o.mSunAngularRadius);
  cs::core::Settings::serialize(j, "sunIlluminance", o.mSunIlluminance);
  cs::core::Settings::serialize(j, "scatteringTextureNuSize", o.mScatteringTextureNuSize);
  cs::core::Settings::serialize(j, "maxSunZenithAngle", o.mMaxSunZenithAngle);
  cs::core::Settings::serialize(j, "refraction", o.mRefraction);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
