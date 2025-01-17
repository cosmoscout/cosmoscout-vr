////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_OPERATIONS_METADATA
#define CSL_OGC_OWS_OPERATIONS_METADATA

#include "19115subset.hpp"
#include "DomainType.hpp"

#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::ows {

struct HTTPType;
struct OperationType;
struct RequestMethodType;

struct CSL_OGC_EXPORT OperationsMetadataType {
  // child elements
  std::vector<OperationType> operations;
  std::vector<DomainType>    parameters;
  std::vector<DomainType>    constraints;
  std::optional<std::any>    extendedCapabilities;
};

struct CSL_OGC_EXPORT OperationType {
  // attributes
  std::string name;

  // child elements
  std::vector<std::variant<HTTPType>> dcps;
  std::vector<DomainType>             parameters;
  std::vector<DomainType>             constraints;
  std::vector<MetadataType>           metadata;
};

struct CSL_OGC_EXPORT HTTPType {
  // child elements
  std::vector<RequestMethodType> gets;
  std::vector<RequestMethodType> posts;
};

struct CSL_OGC_EXPORT RequestMethodType final : OnlineResourceType {
  // child elements
  std::vector<DomainType> constraints;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_OPERATIONS_METADATA
