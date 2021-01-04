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
    : mUrl(url)
    , mTitle(parseTitle())
    , mRootLayer(parseRootLayer()) {
  mRootLayer.getRequestableLayers(mRequestableLayers);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapService::getUrl() const {
  return mUrl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapService::getTitle() const {
  return mTitle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<WebMapLayer> WebMapService::getLayers() const {
  return mRequestableLayers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<WebMapLayer> WebMapService::getLayer(std::string name) const {
  std::vector<WebMapLayer> layers = getLayers();
  auto                     layer  = std::find_if(
      layers.begin(), layers.end(), [&name](WebMapLayer l) { return l.getName() == name; });
  if (layer == layers.end()) {
    return {};
  } else {
    return *layer;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaXML::TiXmlElement* WebMapService::getCapabilities() {
  if (!mDoc.has_value()) {
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
    mDoc = doc;
  }
  return mDoc->FirstChildElement("WMS_Capabilities");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapLayer WebMapService::parseRootLayer() {
  VistaXML::TiXmlHandle   capabilityHandle(getCapabilities());
  VistaXML::TiXmlElement* root =
      capabilityHandle.FirstChildElement("Capability").FirstChildElement("Layer").ToElement();

  WebMapLayer::Settings settings;

  // Set default attribution to contact person if given
  VistaXML::TiXmlElement* contactPerson = capabilityHandle.FirstChildElement("Service")
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

  return WebMapLayer(root, settings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapService::parseTitle() {
  VistaXML::TiXmlHandle capabilityHandle(getCapabilities());
  VistaXML::TiXmlText*  serviceTitle = capabilityHandle.FirstChildElement("Service")
                                          .FirstChildElement("Title")
                                          .FirstChild()
                                          .ToText();
  return serviceTitle->ValueStr();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
