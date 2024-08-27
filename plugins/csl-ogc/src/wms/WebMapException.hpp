////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_WEB_MAP_EXCEPTION_HPP
#define CSL_OGC_WEB_MAP_EXCEPTION_HPP

#include "../common/OGCException.hpp"
#include "../common/OGCExceptionReport.hpp"

#include <VistaTools/tinyXML/tinyxml.h>

#include <string>
#include <vector>

namespace csl::ogc {

/// Class to store a single WMS exception.
class WebMapException : public OGCException {
 public:
  explicit WebMapException(VistaXML::TiXmlElement* element);
};

/// Class to store a collection of WMS exceptions.
class WebMapExceptionReport : public OGCExceptionReport {
 public:
  /// Construct a WebMapExceptionReport from a XML document.
  explicit WebMapExceptionReport(VistaXML::TiXmlDocument const& doc);

  /// Construct a WebMapExceptionReport from a string containing a XML document.
  explicit WebMapExceptionReport(std::string const& xml);

 private:
  std::vector<std::unique_ptr<OGCException>> parseExceptions(VistaXML::TiXmlDocument doc) const;
};

} // namespace csl::ogc

#endif // CSL_OGC_WEB_MAP_EXCEPTION_HPP