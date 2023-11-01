////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WebMapException.hpp"

#include "../common/utils.hpp"

namespace csl::ogc {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapException::WebMapException(VistaXML::TiXmlElement* element)
    : OGCException(utils::getAttribute<std::string>(element, "code").value_or("None"),
          utils::getElementValue<std::string>(element).value_or("No description given.")) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapExceptionReport::WebMapExceptionReport(VistaXML::TiXmlDocument const& doc)
    : OGCExceptionReport(std::move(parseExceptions(doc))) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapExceptionReport::WebMapExceptionReport(std::string const& xml)
    : WebMapExceptionReport(parseXml(xml)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::unique_ptr<OGCException>> WebMapExceptionReport::parseExceptions(
    VistaXML::TiXmlDocument doc) const {
  VistaXML::TiXmlElement* root = doc.FirstChildElement("ServiceExceptionReport");
  if (root == nullptr) {
    return {};
  }

  std::vector<std::unique_ptr<OGCException>> exceptions{};
  for (VistaXML::TiXmlElement* exceptionElement = root->FirstChildElement("ServiceException");
       exceptionElement != nullptr;
       exceptionElement = exceptionElement->NextSiblingElement("ServiceException")) {
    exceptions.push_back(std::make_unique<WebMapException>(exceptionElement));
  }

  return exceptions;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::ogc
