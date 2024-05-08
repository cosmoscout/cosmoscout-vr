////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WebCoverageException.hpp"

#include "../common/utils.hpp"

namespace csl::ogc {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverageException::WebCoverageException(VistaXML::TiXmlElement* element)
    : OGCException(utils::getAttribute<std::string>(element, "exceptionCode").value_or("None"),
          utils::getElementValue<std::string>(element->FirstChildElement("ows:ExceptionText"))
              .value_or("No description given.")) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverageExceptionReport::WebCoverageExceptionReport(VistaXML::TiXmlDocument const& doc)
    : OGCExceptionReport(parseExceptions(doc)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverageExceptionReport::WebCoverageExceptionReport(std::string const& xml)
    : WebCoverageExceptionReport(parseXml(xml)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::unique_ptr<OGCException>> WebCoverageExceptionReport::parseExceptions(
    VistaXML::TiXmlDocument doc) const {
  VistaXML::TiXmlElement* root = doc.FirstChildElement("ows:ExceptionReport");
  if (root == nullptr) {
    return {};
  }

  std::vector<std::unique_ptr<OGCException>> exceptions{};
  for (VistaXML::TiXmlElement* exceptionElement = root->FirstChildElement("ows:Exception");
       exceptionElement != nullptr;
       exceptionElement = exceptionElement->NextSiblingElement("ows:Exception")) {
    exceptions.push_back(std::make_unique<WebCoverageException>(exceptionElement));
  }

  return exceptions;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::ogc
