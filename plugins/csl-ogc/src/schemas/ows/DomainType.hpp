////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_DOMAIN_TYPE
#define CSL_OGC_OWS_DOMAIN_TYPE

#include "Common.hpp"

#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

#include <any>
#include <variant>

namespace ogc::schemas::ows {

struct ReferenceSystem;
struct UOM;
struct RangeType;
struct ValuesReferenceType;
struct UnNamedDomainType;
struct DomainMetadataType;

using AllowedValues = std::vector<std::variant<std::string, RangeType>>;

struct CSL_OGC_EXPORT DomainType final : UnNamedDomainType {
  // attributes
  std::string name;
};

struct CSL_OGC_EXPORT UnNamedDomainType {
  // child elements
  std::variant<std::any, std::monostate, AllowedValues, ValuesReferenceType> possibleValues;

  std::optional<std::string>                        defaultValue;
  std::optional<DomainMetadataType>                 meaning;
  std::optional<DomainMetadataType>                 dataType;
  std::optional<std::variant<UOM, ReferenceSystem>> valuesUnit;
  std::vector<MetadataType>                         metaData;

  virtual ~UnNamedDomainType() = default;
};

struct CSL_OGC_EXPORT ValuesReferenceType {
  // attributes
  std::string reference;

  // content
  std::string content;
};

enum class RangeClosureType {
  CLOSED,
  OPEN,
  OPEN_CLOSED,
  CLOSED_OPEN,
};

struct CSL_OGC_EXPORT RangeType {
  // attributes
  std::optional<RangeClosureType> rangeClosure;

  // child elements
  std::optional<std::string> minimumValue;
  std::optional<std::string> maximumValue;
  std::optional<std::string> spacing;
};

struct CSL_OGC_EXPORT DomainMetadataType {
  // attributes
  std::optional<std::string> reference;

  // content
  std::string content;

  ~DomainMetadataType() = default;
};

struct CSL_OGC_EXPORT UOM : DomainMetadataType {};

struct CSL_OGC_EXPORT ReferenceSystem : DomainMetadataType {};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_DOMAIN_TYPE
