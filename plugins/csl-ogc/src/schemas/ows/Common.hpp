////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_COMMON
#define CSL_OGC_OWS_COMMON

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../xlink/XLink.hpp"

#include "csl_ogc_export.hpp"

namespace ogc::schemas::ows {

using MimeType    = std::string;
using VersionType = std::string;
using PositionType   = std::vector<double>;

struct AbstractMetadataType {
  virtual ~AbstractMetadataType() = default;
};

struct MetadataType {
  // attributes
  xlink::SimpleAttrsGroup simpleAttributes;

  // child elements
  std::unique_ptr<AbstractMetadataType> abstractMetadata;

  virtual ~MetadataType() = default;
};

struct CSL_OGC_EXPORT BoundingBoxType {
  // attributes
  std::optional<std::string> crs;
  std::optional<uint64_t>    dimensions;

  // child elements
  PositionType lowerCorner;
  PositionType upperCorner;

  virtual ~BoundingBoxType() = default;
};

struct CSL_OGC_EXPORT WGS84BoundingBoxType final : BoundingBoxType {
  WGS84BoundingBoxType()
      : BoundingBoxType() {
    dimensions = 2;
    crs = "urn:ogc:def:crs:OGC:2:84";
  }
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_COMMON