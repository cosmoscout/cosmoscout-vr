////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_ADDITIONAL_PARAMETERS
#define CSL_OGC_OWS_ADDITIONAL_PARAMETERS

#include "19115subset.hpp"
#include "Common.hpp"

#include <vector>

#include "csl_ogc_export.hpp"

#include <any>

namespace ogc::schemas::ows {

struct CSL_OGC_EXPORT AdditionalParameterType final : AbstractMetadataType {
  CodeType name;
  std::vector<std::any> values;
};

struct CSL_OGC_EXPORT AdditionalParametersBaseType : MetadataType {
  AdditionalParametersBaseType() {
    abstractMetadata = std::make_unique<AdditionalParameterType>();
  }

  AdditionalParameterType* getAdditionalParameter() const {
    return dynamic_cast<AdditionalParameterType*>(abstractMetadata.get());
  }

  ~AdditionalParametersBaseType() override = default;
};

struct CSL_OGC_EXPORT AdditionalParametersType final : AdditionalParametersBaseType {
  // child element
  std::vector<AdditionalParameterType> additionalParameters;
};

struct CSL_OGC_EXPORT NilValueType final : CodeType {
  // attributes
  std::optional<std::string> nilReason;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_ADDITIONAL_PARAMETERS