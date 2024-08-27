////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_WEB_COVERAGE_EXCEPTION_HPP
#define CSL_OGC_WEB_COVERAGE_EXCEPTION_HPP

#include "../common/OGCException.hpp"
#include "../common/OGCExceptionReport.hpp"

#include <VistaTools/tinyXML/tinyxml.h>

#include <string>
#include <vector>

namespace csl::ogc {

/// Class to store a single WCS exception.
class WebCoverageException : public OGCException {
 public:
  explicit WebCoverageException(VistaXML::TiXmlElement* element);
};

/// Class to store a collection of WCS exceptions.
class WebCoverageExceptionReport : public OGCExceptionReport {
 public:
  /// Construct a WebCoverageExceptionReport from a XML document.
  explicit WebCoverageExceptionReport(VistaXML::TiXmlDocument const& doc);

  /// Construct a WebCoverageExceptionReport from a string containing a XML document.
  explicit WebCoverageExceptionReport(std::string const& xml);

 private:
  std::vector<std::unique_ptr<OGCException>> parseExceptions(VistaXML::TiXmlDocument doc) const;
};

} // namespace csl::ogc

#endif // CSL_OGC_WEB_COVERAGE_EXCEPTION_HPP