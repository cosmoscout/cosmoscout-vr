////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_GET_CAPABILITIES
#define CSL_OGC_OWS_GET_CAPABILITIES

#include "OperationsMetadataType.hpp"
#include "ServiceIdentificationType.hpp"
#include "ServiceProviderType.hpp"

#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::ows {

struct AcceptVersionsType;
struct SectionsType;
struct AcceptFormatsType;
struct LanguagesType;

using ServiceType = std::string;
using UpdateSequenceType = std::string;

struct CSL_OGC_EXPORT CapabilitiesBaseType {
  // attributes
  VersionType                   version;
  std::optional<UpdateSequenceType> updateSequence;

  // child elements
  std::optional<ServiceIdentificationType> serviceIdentification;
  std::optional<ServiceProviderType>       serviceProvider;
  std::optional<OperationsMetadataType>    operationsMetadata;
  std::optional<LanguagesType>             languages;

  virtual ~CapabilitiesBaseType() = default;
};

struct CSL_OGC_EXPORT GetCapabilitiesType {
  // attributes
  std::optional<UpdateSequenceType> updateSequence;

  // child elements
  std::optional<AcceptVersionsType> acceptVersions;
  std::optional<SectionsType>       sections;
  std::optional<AcceptFormatsType>  acceptFormats;
  std::optional<LanguagesType>      acceptLanguages;

  virtual ~GetCapabilitiesType() = default;
};

struct CSL_OGC_EXPORT LanguagesType {
  std::vector<std::string> languages;
};

struct CSL_OGC_EXPORT AcceptVersionsType {
  std::vector<VersionType> versions;
};

struct CSL_OGC_EXPORT SectionsType {
  std::vector<std::string> sections;
};

struct CSL_OGC_EXPORT AcceptFormatsType {
  std::vector<MimeType> mimeFormats;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_GET_CAPABILITIES