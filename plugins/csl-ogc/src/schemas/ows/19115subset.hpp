////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_19115_SUBSET
#define CSL_OGC_OWS_19115_SUBSET

#include <optional>
#include <string>
#include <vector>

#include "../xlink/XLink.hpp"

#include "csl_ogc_export.hpp"

namespace ogc::schemas::ows {

struct OnlineResourceType;
struct AddressType;
struct TelephoneType;
struct ContactType;
struct CodeType;

struct CSL_OGC_EXPORT LanguageStringType {
  // attributes
  std::optional<std::string> lang;

  // content
  std::string content;
};

struct CSL_OGC_EXPORT KeywordsType {
  // child elements
  std::vector<LanguageStringType> keywords;
  std::optional<CodeType>         type;
};

struct CSL_OGC_EXPORT CodeType {
  // attributes
  std::optional<std::string> codeSpace;

  // content
  std::string content;

  virtual ~CodeType() = default;
};

struct CSL_OGC_EXPORT ResponsiblePartyType {
  // child elements
  std::optional<std::string> individualName;
  std::optional<std::string> organizationName;
  std::optional<std::string> positionName;
  std::optional<ContactType> contactInfo;
  CodeType                   role;
};

struct CSL_OGC_EXPORT ResponsiblePartySubsetType {
  // child elements
  std::optional<std::string> individualName;
  std::optional<std::string> positionName;
  std::optional<ContactType> contactInfo;
  std::optional<CodeType>    role;
};

struct CSL_OGC_EXPORT ContactType {
  // child elements
  std::optional<TelephoneType>      phone;
  std::optional<AddressType>        address;
  std::optional<OnlineResourceType> onlineResource;
  std::optional<std::string>        hoursOfService;
  std::optional<std::string>        contactInstructions;
};

struct CSL_OGC_EXPORT TelephoneType {
  // child elements
  std::vector<std::string> voiceNumbers;
  std::vector<std::string> facsimileNumbers;
};

struct CSL_OGC_EXPORT AddressType {
  // child elements
  std::vector<std::string>   deliveryPoint;
  std::optional<std::string> city;
  std::optional<std::string> administrativeArea;
  std::optional<std::string> postalCode;
  std::optional<std::string> country;
  std::vector<std::string>   electronicMailAddresses;
};

struct CSL_OGC_EXPORT OnlineResourceType {
  // attributes
  xlink::SimpleAttrsGroup simpleAttributes;

  virtual ~OnlineResourceType() = default;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_19115_SUBSET