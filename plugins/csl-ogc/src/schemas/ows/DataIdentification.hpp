////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_DATA_IDENTIFICATION
#define CSL_OGC_OWS_DATA_IDENTIFICATION

#include "19115subset.hpp"
#include "Common.hpp"

#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::ows {

struct CSL_OGC_EXPORT DescriptionType {
  // child elements
  std::vector<std::string>  titles;
  std::vector<std::string>  abstracts;
  std::vector<KeywordsType> keywords;

  virtual ~DescriptionType() = default;
};

struct CSL_OGC_EXPORT BasicIdentificationType : DescriptionType {
  // child elements
  std::optional<CodeType>   identifier;
  std::vector<MetadataType> metaData;

  ~BasicIdentificationType() override = default;
};

struct CSL_OGC_EXPORT IdentificationType final : BasicIdentificationType {
  // child elements
  std::vector<BoundingBoxType> boundingBoxes;
  std::vector<MimeType>        outputFormats;
  std::vector<std::string>     availableCRSs;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_DATA_IDENTIFICATION