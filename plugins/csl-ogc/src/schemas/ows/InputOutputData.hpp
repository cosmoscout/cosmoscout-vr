////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_INPUT_OUTPUT_DATA
#define CSL_OGC_OWS_INPUT_OUTPUT_DATA

#include "Manifest.hpp"

#include <any>
#include <variant>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::ows {

struct CSL_OGC_EXPORT ServiceReferenceType final : ReferenceType {
  // child elements
  std::variant<std::any, std::string> requestMessage;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_INPUT_OUTPUT_DATA