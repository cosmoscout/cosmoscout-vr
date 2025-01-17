////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_GET_RESOURCE_BY_ID
#define CSL_OGC_OWS_GET_RESOURCE_BY_ID

#include "Common.hpp"
#include "GetCapabilitiesType.hpp"

#include <vector>

#include "csl_ogc_export.hpp"

#include <optional>
#include <string>

namespace ogc::schemas::ows {

struct CSL_OGC_EXPORT GetResourceByIdType {
  // attributes
  ServiceType service;
  VersionType version;

  // child elements
  std::vector<std::string> resourceIDs;
  std::optional<MimeType>  outputFormat;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_GET_RESOURCE_BY_ID