////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_WCS_GET_CAPABILITIES
#define CSL_OGC_WCS_GET_CAPABILITIES

#include "Common.hpp"

#include <optional>
#include <string>
#include <vector>

#include "../ows/Contents.hpp"
#include "../ows/GetCapabilitiesType.hpp"

#include "csl_ogc_export.hpp"

namespace ogc::schemas::wcs {

struct ContentsType;
struct CoverageSummaryType;

struct CSL_OGC_EXPORT GetCapabilitiesType final : ows::GetCapabilitiesType {
  // attributes
  ows::ServiceType service = "WCS";
};

struct CSL_OGC_EXPORT CapabilitiesType final : ows::CapabilitiesBaseType {
  // child elements
  std::optional<ServiceMetadataType> TODO;
  std::optional<ContentsType>        contents;
};

struct CSL_OGC_EXPORT ContentsType final : ows::ContentsBaseType {
  // child elements
  std::vector<CoverageSummaryType> coverageSummary;
  std::optional<ExtensionType>     extension;
};

struct CSL_OGC_EXPORT CoverageSummaryType final : ows::DescriptionType {
  // child elements
  std::vector<ows::WGS84BoundingBoxType>   wgs84BoundingBoxes;
  std::string                              coverageId;
  std::string                              coverageSubtype;
  std::optional<CoverageSubtypeParentType> coverageSubtypeParent;
  std::vector<ows::BoundingBoxType>        boundingBoxes;
  std::vector<ows::MetadataType>           metadata;
};

} // namespace ogc::schemas::wcs

#endif // CSL_OGC_WCS_GET_CAPABILITIES