////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_WCS_COMMON
#define CSL_OGC_WCS_COMMON

#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

#include <any>

namespace ogc::schemas::wcs {

struct ExtensionType;
struct ServiceMetadataType;
struct OfferedCoverageType;
struct ServiceParametersType;
struct CoverageSubtypeParentType;

using VersionStringType = std::string;

struct CSL_OGC_EXPORT RequestBaseType {
  // attributes
  std::string       service = "WCS";
  VersionStringType version;

  // child elements
  std::optional<ExtensionType> extension;
};

struct CSL_OGC_EXPORT CoverageOfferingsType {
  // child elements
  ServiceMetadataType              serviceMetadata;
  std::vector<OfferedCoverageType> offeredCoverages;
};

struct CSL_OGC_EXPORT ServiceMetadataType {
  // child elements
  std::vector<std::string>     formatsSupported;
  std::optional<ExtensionType> extension;
};

struct CSL_OGC_EXPORT OfferedCoverageType {
  // child elements
  // gmlcov::AbstractCoverage TODO;
  ServiceParametersType serviceParameters;
};

struct CSL_OGC_EXPORT ServiceParametersType {
  // child elements
  std::string                              coverageSubtype;
  std::optional<CoverageSubtypeParentType> coverageSubtypeParent;
  std::string                              nativeFormat;
  std::optional<ExtensionType>             extension;
};

struct CSL_OGC_EXPORT CoverageSubtypeParentType {
  // child elements
  std::string                              coverageSubtype;
  std::optional<CoverageSubtypeParentType> coverageSubtypeParent;
};

struct CSL_OGC_EXPORT ExtensionType {
  // child elements
  std::vector<std::any> children;
};

} // namespace ogc::schemas::wcs

#endif // CSL_OGC_WCS_COMMON