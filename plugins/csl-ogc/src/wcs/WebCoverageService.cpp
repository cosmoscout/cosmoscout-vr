////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WebCoverageService.hpp"
#include "../logger.hpp"
#include "WebCoverageException.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../common/utils.hpp"

#include <regex>

#include <boost/filesystem.hpp>

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

namespace csl::ogc {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverageService::WebCoverageService(std::string url, CacheMode cacheMode, std::string cacheDir)
    : WebServiceBase(
          std::move(url), cacheMode, std::move(cacheDir), "WCS", "2.0.1", {"wcs:Capabilities"}) {
  setTitle(parseTitle());
  parseCoverages();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<WebCoverage> const& WebCoverageService::getCoverages() const {
  return mRequestableCoverages;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<WebCoverage> WebCoverageService::getCoverage(std::string const& titleOrId) const {
  std::vector<WebCoverage> coverages = getCoverages();

  auto coverage = std::find_if(
      coverages.begin(), coverages.end(), [titleOrId](WebCoverage const& capabilityCoverage) {
        return capabilityCoverage.getTitle() == titleOrId ||
               capabilityCoverage.getId() == titleOrId;
      });

  if (coverage == coverages.end()) {
    return {};
  }

  return *coverage;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<OGCExceptionReport> WebCoverageService::createExceptionReport(
    VistaXML::TiXmlDocument const& doc) const {
  return std::make_unique<WebCoverageExceptionReport>(doc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebCoverageService::parseCoverages() {
  VistaXML::TiXmlHandle   capabilityHandle(getCapabilities());
  VistaXML::TiXmlElement* contents = capabilityHandle.FirstChildElement("wcs:Contents").ToElement();

  WebCoverage::Settings settings;

  // Set default attribution to contact person if given
  VistaXML::TiXmlElement* contactPerson =
      capabilityHandle.FirstChildElement("ows:ServiceProvider").ToElement();

  if (contactPerson != nullptr) {
    std::optional<std::string> organization =
        utils::getElementValue<std::string>(contactPerson, {"ows:ProviderName"});
    if (organization.has_value()) {
      settings.mAttribution = organization.value();
    }
  }

  for (auto* coverage = contents->FirstChild("wcs:CoverageSummary"); coverage;
       coverage       = coverage->NextSibling()) {
    if (!coverage->NoChildren()) {
      try {
        mRequestableCoverages.emplace_back(WebCoverage(coverage->ToElement(), settings, getUrl()));
      } catch (std::exception const& e) { logger().trace(e.what()); }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebCoverageService::parseTitle() {
  return utils::getElementValue<std::string>(
      getCapabilities(), {"ows:ServiceIdentification", "ows:Title"})
      .value_or("Untitled");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::ogc