////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "BRDFFactory.hpp"

namespace cs::graphics {

char const* BRDF_SNIPPED_TEMPLATE = R"(
$BRDF_HDR
$BRDF_NON_HDR
)";

std::string BRDFFactory::getBRDFSnipped() const {
  return "";
}

} // namespace cs::graphics
