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
    }
  }

  std::optional<double> minLon = utils::optstod(
      utils::getElementText(element, {"EX_GeographicBoundingBox", "westBoundLongitude"}));
  std::optional<double> maxLon = utils::optstod(
      utils::getElementText(element, {"EX_GeographicBoundingBox", "eastBoundLongitude"}));
  std::optional<double> minLat = utils::optstod(
      utils::getElementText(element, {"EX_GeographicBoundingBox", "southBoundLongitude"}));
  std::optional<double> maxLat = utils::optstod(
      utils::getElementText(element, {"EX_GeographicBoundingBox", "northBoundLongitude"}));

  for (VistaXML::TiXmlElement* boundingBoxElement = element->FirstChildElement("BoundingBox");
       boundingBoxElement;
       boundingBoxElement = boundingBoxElement->NextSiblingElement("BoundingBox")) {
    std::string crs =
        utils::getAttribute<std::string>(boundingBoxElement, "CRS").value_or("No CRS");
    if (crs == "CRS:84") {
      minLon = utils::getAttribute<double>(boundingBoxElement, "minx");
      maxLon = utils::getAttribute<double>(boundingBoxElement, "maxx");
      minLat = utils::getAttribute<double>(boundingBoxElement, "miny");
      maxLat = utils::getAttribute<double>(boundingBoxElement, "maxy");
    } else if (crs == "EPSG:4326") {
      minLon = utils::getAttribute<double>(boundingBoxElement, "miny");
      maxLon = utils::getAttribute<double>(boundingBoxElement, "maxy");
      minLat = utils::getAttribute<double>(boundingBoxElement, "minx");
      maxLat = utils::getAttribute<double>(boundingBoxElement, "maxx");
    }
  }
  utils::setOrKeep(mSettings.mLonRange[0], minLon);
  utils::setOrKeep(mSettings.mLonRange[1], maxLon);
  utils::setOrKeep(mSettings.mLatRange[0], minLat);
  utils::setOrKeep(mSettings.mLatRange[1], maxLat);

  for (VistaXML::TiXmlElement* styleElement = element->FirstChildElement("Style"); styleElement;
       styleElement                         = styleElement->NextSiblingElement("Style")) {
    mSettings.mStyles.emplace_back(styleElement);
  }

  // TODO Other dimensions?
  // TODO CRS

  for (VistaXML::TiXmlElement* layerElement = element->FirstChildElement("Layer"); layerElement;
       layerElement                         = layerElement->NextSiblingElement("Layer")) {
    mSubLayers.push_back(WebMapLayer(layerElement, mSettings));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapLayer::getTitle() const {
  return mTitle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapLayer::getName() const {
  return mName.value_or("");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapLayer::Settings WebMapLayer::getSettings() const {
  return mSettings;
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

WebMapLayer::Style::Style(VistaXML::TiXmlElement* element)
    : mName(utils::getElementText(element, {"Name"}).value())
    , mTitle(utils::getElementText(element, {"Title"}).value_or(mName))
    , mLegendUrl(getLegendUrl(element)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::string> WebMapLayer::Style::getLegendUrl(VistaXML::TiXmlElement* element) {
  VistaXML::TiXmlHandle   handle(element);
  VistaXML::TiXmlElement* resource =
      handle.FirstChildElement("LegendURL").FirstChildElement("OnlineResource").ToElement();
  if (resource != nullptr) {
    return utils::getAttribute<std::string>(resource, "xlink:href");
  }
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
