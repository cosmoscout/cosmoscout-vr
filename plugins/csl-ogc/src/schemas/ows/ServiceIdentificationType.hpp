////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_SERVICE_IDENTIFICATION
#define CSL_OGC_OWS_SERVICE_IDENTIFICATION

#include "DataIdentification.hpp"

#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::ows {

struct CSL_OGC_EXPORT ServiceIdentificationType final : DescriptionType {
  // child elements
  CodeType                   serviceType;
  std::vector<VersionType>   versions;
  std::vector<std::string>   profiles;
  std::optional<std::string> fees;
  std::vector<std::string>   accessConstraints;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_SERVICE_IDENTIFICATION