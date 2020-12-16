////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebMapService.hpp"
#include "logger.hpp"

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

namespace csp::wmsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapService::WebMapService(std::string url)
    : mUrl(url) {
  VistaXML::TiXmlDocument capabilities = getCapabilities();
  VistaXML::TiXmlHandle   capabilityHandle(&capabilities);
  VistaXML::TiXmlElement* root = capabilityHandle.FirstChildElement("WMS_Capabilities")
                                     .FirstChildElement("Capability")
                                     .FirstChildElement("Layer")
                                     .ToElement();
  WebMapLayer::Settings settings;

  // Set default attribution to contact person if given
  VistaXML::TiXmlElement* contactPerson = capabilityHandle.FirstChildElement("WMS_Capabilities")
                                              .FirstChildElement("Service")
                                              .FirstChildElement("ContactInformation")
                                              .FirstChildElement("ContactPersonPrimary")
                                              .ToElement();
  if (contactPerson != nullptr) {
    std::stringstream    contact;
    VistaXML::TiXmlNode* organization =
        contactPerson->FirstChildElement("ContactOrganization")->FirstChild();
    VistaXML::TiXmlNode* person = contactPerson->FirstChildElement("ContactPerson")->FirstChild();
    if (person != nullptr) {
      contact << person->ValueStr();
    }
    if (person != nullptr && organization != nullptr) {
      contact << ", ";
    }
    if (organization != nullptr) {
      contact << organization->ValueStr();
    }
    settings.mAttribution = contact.str();
  }

  mRootLayer = std::make_unique<WebMapLayer>(root, settings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<WebMapLayer> WebMapService::getLayers() {
  std::vector<WebMapLayer> layers;
  mRootLayer->getRequestableLayers(layers);
  return layers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaXML::TiXmlDocument WebMapService::getCapabilities() {
  std::stringstream urlStream;
  urlStream << mUrl;
  urlStream << "?SERVICE=WMS&version=1.3.0&REQUEST=GetCapabilities";
  const std::string urlString = urlStream.str();

  std::stringstream xmlStream;
  curlpp::Easy      request;
  request.setOpt(curlpp::options::Url(urlString));
  request.setOpt(curlpp::options::WriteStream(&xmlStream));
  request.setOpt(curlpp::options::NoSignal(true));
  request.setOpt(curlpp::options::SslVerifyPeer(false));

  try {
    request.perform();
  } catch (std::exception& e) {
    logger().error(
        "Failed to perform WMS Capabilities request: '{}'! Exception: '{}'", urlString, e.what());
  }

  const std::string       xmlString = xmlStream.str();
  VistaXML::TiXmlDocument doc;
  doc.Parse(xmlString.c_str());
  if (doc.Error()) {
    logger().error("Parsing failed with '{}'", doc.ErrorDesc());
  } else {
    logger().trace("Successfully parsed xml");
  }
  return doc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
