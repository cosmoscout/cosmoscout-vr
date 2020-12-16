////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebMapLayer.hpp"
#include "logger.hpp"
#include "utils.hpp"

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

namespace csp::wmsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapLayer::WebMapLayer(VistaXML::TiXmlElement* element, Settings settings)
    : mSettings(settings) {
  VistaXML::TiXmlHandle elementHandle(element);

  VistaXML::TiXmlElement* titleElement = element->FirstChildElement("Title");
  if (titleElement == nullptr) {
    // TODO Custom Exception
    throw 0;
  }
  mTitle = titleElement->FirstChild()->ValueStr();

  VistaXML::TiXmlElement* nameElement = element->FirstChildElement("Name");
  if (nameElement != nullptr) {
    mName = nameElement->FirstChild()->ValueStr();
  }

  utils::setOrKeep(mSettings.mOpaque, utils::getBoolAttribute(element, "opaque"));
  utils::setOrKeep(mSettings.mNoSubsets, utils::getBoolAttribute(element, "noSubsets"));

  utils::setOrKeep(mSettings.mFixedWidth, utils::getSizeAttribute(element, "fixedWidth"));
  utils::setOrKeep(mSettings.mFixedHeight, utils::getSizeAttribute(element, "fixedHeight"));

  utils::setOrKeep(
      mSettings.mAttribution, utils::getElementText(element, {"Attribution", "Title"}));

  for (VistaXML::TiXmlElement* dimensionElement = element->FirstChildElement("Dimension");
       dimensionElement; dimensionElement = dimensionElement->NextSiblingElement("Dimension")) {
    if (utils::getAttribute<std::string>(dimensionElement, "name").value() == "time") {
      utils::setOrKeep(mSettings.mTime, utils::getElementText(dimensionElement, {}));
      logger().trace("Time: {}", mSettings.mTime.value());
    }
  }

  // TODO Bounding Box
  // TODO Other dimensions?
  // TODO Styles + Legends
  // TODO CRS

  for (VistaXML::TiXmlElement* layerElement = element->FirstChildElement("Layer"); layerElement;
       layerElement                         = layerElement->NextSiblingElement("Layer")) {
    mSubLayers.push_back(WebMapLayer(layerElement, mSettings));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapLayer::getTitle() {
  return mTitle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebMapLayer::isRequestable() {
  return mName.has_value();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebMapLayer::getRequestableLayers(std::vector<WebMapLayer>& layers) {
  if (isRequestable()) {
    layers.push_back(*this);
  }
  for (WebMapLayer sub : mSubLayers) {
    sub.getRequestableLayers(layers);
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
