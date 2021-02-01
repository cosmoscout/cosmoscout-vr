////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebMapLayer.hpp"
#include "logger.hpp"
#include "utils.hpp"

#include "../../../src/cs-utils/utils.hpp"

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

namespace csp::wmsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapLayer::WebMapLayer(VistaXML::TiXmlElement* element, Settings settings)
    : mSettings(settings) {
  VistaXML::TiXmlHandle elementHandle(element);

  VistaXML::TiXmlElement* titleElement = element->FirstChildElement("Title");
  if (titleElement == nullptr) {
    throw std::runtime_error("No title found for Layer.");
  }
  mTitle = titleElement->FirstChild()->ValueStr();

  mName = utils::getElementText(element, {"Name"});
  mAbstract = utils::getElementText(element, {"Abstract"});

  utils::setOrKeep(mSettings.mOpaque, utils::getAttribute<bool>(element, "opaque"));
  utils::setOrKeep(mSettings.mNoSubsets, utils::getAttribute<bool>(element, "noSubsets"));

  utils::setOrKeep(mSettings.mFixedWidth, utils::getSizeAttribute(element, "fixedWidth"));
  utils::setOrKeep(mSettings.mFixedHeight, utils::getSizeAttribute(element, "fixedHeight"));

  utils::setOrKeep(
      mSettings.mAttribution, utils::getElementText(element, {"Attribution", "Title"}));

  for (VistaXML::TiXmlElement* dimensionElement = element->FirstChildElement("Dimension");
       dimensionElement; dimensionElement = dimensionElement->NextSiblingElement("Dimension")) {
    if (utils::getAttribute<std::string>(dimensionElement, "name").value() == "time") {
      std::optional<std::string> timeString;
      utils::setOrKeep(timeString, utils::getElementText(dimensionElement, {}));
      if (timeString.has_value()) {
        mSettings.mTimeIntervals.clear();
        try {
          utils::parseIsoString(timeString.value(), mSettings.mTimeIntervals);
        } catch (std::exception const& e) {
          logger().warn("Failed to parse Iso String '{}' for layer '{}': '{}'. No time dependent "
                        "data will be available for this layer!",
              timeString.value(), mTitle, e.what());
        }
      }
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
  utils::setOrKeep(mSettings.mBounds.mMinLon, minLon);
  utils::setOrKeep(mSettings.mBounds.mMaxLon, maxLon);
  utils::setOrKeep(mSettings.mBounds.mMinLat, minLat);
  utils::setOrKeep(mSettings.mBounds.mMaxLat, maxLat);

  for (VistaXML::TiXmlElement* styleElement = element->FirstChildElement("Style"); styleElement;
       styleElement                         = styleElement->NextSiblingElement("Style")) {
    mSettings.mStyles.emplace_back(styleElement);
  }

  for (VistaXML::TiXmlElement* crsElement = element->FirstChildElement("CRS"); crsElement;
       crsElement                         = crsElement->NextSiblingElement("CRS")) {
    std::optional<std::string> crs = utils::getElementText(crsElement, {});
    if (crs.has_value() && !cs::utils::contains(mSettings.mCrs, crs.value())) {
      mSettings.mCrs.push_back(crs.value());
    }
  }

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

std::optional<std::string> WebMapLayer::getAbstract() const {
  return mAbstract;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapLayer::Settings WebMapLayer::getSettings() const {
  return mSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebMapLayer::isRequestable() const {
  // According to 7.2.4.6.3 of the WMS 1.3.0 implementation specification maps may be requested for
  // all layers with a given name.
  return mName.has_value();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<WebMapLayer> WebMapLayer::getAllLayers() const {
  return mSubLayers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebMapLayer::getRequestableLayers(std::vector<WebMapLayer>& layers) const {
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
    , mTitle(utils::getElementText(element, {"Title"}).value_or("Untitled"))
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
