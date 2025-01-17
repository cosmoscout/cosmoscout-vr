////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_MANIFEST
#define CSL_OGC_OWS_MANIFEST

#include "19115subset.hpp"
#include "Common.hpp"
#include "DataIdentification.hpp"

#include <vector>

#include "csl_ogc_export.hpp"

#include <optional>
#include <string>

namespace ogc::schemas::ows {

struct CSL_OGC_EXPORT AbstractReferenceBaseType {
  // attributes
  std::string                type = "simple";
  std::string                href;
  std::optional<std::string> role;
  std::optional<std::string> arcrole;
  std::optional<std::string> title;
  std::optional<std::string> show;
  std::optional<std::string> actuate;

  virtual ~AbstractReferenceBaseType() = default;
};

struct CSL_OGC_EXPORT ReferenceType : AbstractReferenceBaseType {
  // child elements
  std::optional<CodeType>         identifier;
  std::vector<LanguageStringType> abstracts;
  std::optional<MimeType>         format;
  std::vector<MetadataType>       metadata;

  ~ReferenceType() override = default;
};

struct CSL_OGC_EXPORT ReferenceGroupType final : BasicIdentificationType {
  // child elements
  std::vector<std::unique_ptr<AbstractReferenceBaseType>> references;
};

struct CSL_OGC_EXPORT ManifestType final : BasicIdentificationType {
  // child elements
  std::vector<ReferenceGroupType> referenceGroups;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_MANIFEST