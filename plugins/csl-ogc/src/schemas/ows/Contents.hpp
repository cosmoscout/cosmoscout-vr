////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_CONTENTS
#define CSL_OGC_OWS_CONTENTS

#include "Common.hpp"
#include "DataIdentification.hpp"

#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::ows {

struct DatasetDescriptionSummaryBaseType;

struct CSL_OGC_EXPORT ContentsBaseType {
  // child elements
  std::vector<DatasetDescriptionSummaryBaseType> datasetDescriptionSummaries;
  std::vector<MetadataType>                      otherSources;

  virtual ~ContentsBaseType() = default;
};

struct CSL_OGC_EXPORT DatasetDescriptionSummaryBaseType final : DescriptionType {
  // child elements
  std::vector<WGS84BoundingBoxType>              wgs84BoundingBoxes;
  CodeType                                       identifier;
  std::vector<BoundingBoxType>                   boundingBoxes;
  std::vector<MetadataType>                      metadata;
  std::vector<DatasetDescriptionSummaryBaseType> datasetDescriptionSummaries;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_CONTENTS