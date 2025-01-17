////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_SERVICE_PROVIDER
#define CSL_OGC_OWS_SERVICE_PROVIDER

#include "DataIdentification.hpp"

#include <optional>
#include <string>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::ows {

struct CSL_OGC_EXPORT ServiceProviderType {
  // child elements
  std::string                       providerName;
  std::optional<OnlineResourceType> providerSite;
  ResponsiblePartySubsetType        serviceContact;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_SERVICE_PROVIDER