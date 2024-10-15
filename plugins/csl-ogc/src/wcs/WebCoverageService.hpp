////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_WEB_COVERAGE_SERVICE_HPP
#define CSL_OGC_WEB_COVERAGE_SERVICE_HPP

#include "csl_ogc_export.hpp"

#include "WebCoverage.hpp"

#include "../common/OGCExceptionReport.hpp"
#include "../common/WebServiceBase.hpp"

#include <VistaTools/tinyXML/tinyxml.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace csl::ogc {

class CSL_OGC_EXPORT WebCoverageService : public WebServiceBase {
 public:
  /// Create an object for a WCS accessible at the given URL.
  /// The url string should be the base URL of the WCS without a query string.
  /// cacheMode can be used to control the caching behavior for the capability document.
  /// If caching is activated, cacheDir should be the path to a directory which can be
  /// used for caching.
  WebCoverageService(std::string url, CacheMode cacheMode, std::string cacheDir);

  /// Gets a list of all coverages of the service, for which a coverage can be requested.
  std::vector<WebCoverage> const& getCoverages() const;
  /// Gets the coverage with the given title or coverage id, if one exists.
  /// Returns an empty optional otherwise.
  std::optional<WebCoverage> getCoverage(std::string const& titleOrId) const;

 protected:
  std::unique_ptr<OGCExceptionReport> createExceptionReport(
      VistaXML::TiXmlDocument const& doc) const override;

 private:
  void        parseCoverages();
  std::string parseTitle();

  std::vector<WebCoverage> mRequestableCoverages;
};

} // namespace csl::ogc

#endif // CSL_OGC_WEB_COVERAGE_SERVICE_HPP
