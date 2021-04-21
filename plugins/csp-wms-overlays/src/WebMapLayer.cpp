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
    : mSettings(std::move(settings)) {
  std::optional<std::string> title = utils::getElementValue<std::string>(element, {"Title"});
  if (!title.has_value()) {
    throw std::runtime_error("No title found for Layer.");
  }
  mTitle = title.value();

  mName     = utils::getElementValue<std::string>(element, {"Name"});
  mAbstract = utils::getElementValue<std::string>(element, {"Abstract"});

  utils::setOrKeep(mSettings.mOpaque, utils::getAttribute<bool>(element, "opaque"));
  utils::setOrKeep(mSettings.mNoSubsets, utils::getAttribute<bool>(element, "noSubsets"));

  utils::setOrKeep(mSettings.mFixedWidth, utils::getSizeAttribute(element, "fixedWidth"));
  utils::setOrKeep(mSettings.mFixedHeight, utils::getSizeAttribute(element, "fixedHeight"));

  utils::setOrKeep(mSettings.mAttribution,
      utils::getElementValue<std::string>(element, {"Attribution", "Title"}));

  for (VistaXML::TiXmlElement* dimensionElement = element->FirstChildElement("Dimension");
       dimensionElement; dimensionElement = dimensionElement->NextSiblingElement("Dimension")) {
    if (utils::getAttribute<std::string>(dimensionElement, "name").value() == "time") {
      std::optional<std::string> timeString = utils::getElementValue<std::string>(dimensionElement);
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

  std::optional<double> minLon =
      utils::getElementValue<double>(element, {"EX_GeographicBoundingBox", "westBoundLongitude"});
  std::optional<double> maxLon =
      utils::getElementValue<double>(element, {"EX_GeographicBoundingBox", "eastBoundLongitude"});
  std::optional<double> minLat =
      utils::getElementValue<double>(element, {"EX_GeographicBoundingBox", "southBoundLongitude"});
  std::optional<double> maxLat =
      utils::getElementValue<double>(element, {"EX_GeographicBoundingBox", "northBoundLongitude"});

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

  utils::setOrKeep(
      mSettings.mMinScale, utils::getElementValue<double>(element, {"MinScaleDenominator"}));
  utils::setOrKeep(
      mSettings.mMaxScale, utils::getElementValue<double>(element, {"MaxScaleDenominator"}));

  for (VistaXML::TiXmlElement* styleElement = element->FirstChildElement("Style"); styleElement;
       styleElement                         = styleElement->NextSiblingElement("Style")) {
    mSettings.mStyles.emplace_back(styleElement);
  }

  for (VistaXML::TiXmlElement* crsElement = element->FirstChildElement("CRS"); crsElement;
       crsElement                         = crsElement->NextSiblingElement("CRS")) {
    std::optional<std::string> crs = utils::getElementValue<std::string>(crsElement);
    if (crs.has_value() && !cs::utils::contains(mSettings.mCrs, crs.value())) {
      mSettings.mCrs.push_back(crs.value());
    }
  }

  for (VistaXML::TiXmlElement* layerElement = element->FirstChildElement("Layer"); layerElement;
       layerElement                         = layerElement->NextSiblingElement("Layer")) {
    mSubLayers.emplace_back(layerElement, mSettings);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WebMapLayer::getTitle() const {
  return mTitle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapLayer::getName() const {
  return mName.value_or("");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::string> const& WebMapLayer::getAbstract() const {
  return mAbstract;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapLayer::Settings const& WebMapLayer::getSettings() const {
  return mSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebMapLayer::isRequestable() const {
  // According to 7.2.4.6.3 of the WMS 1.3.0 implementation specification maps may be requested for
  // all layers with a given name.
  return mName.has_value();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<WebMapLayer> const& WebMapLayer::getAllLayers() const {
  return mSubLayers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebMapLayer::getRequestableLayers(std::vector<WebMapLayer>& layers) const {
  if (isRequestable()) {
    layers.push_back(*this);
  }
  for (WebMapLayer const& sub : mSubLayers) {
    sub.getRequestableLayers(layers);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapLayer::Style::Style(VistaXML::TiXmlElement* element)
    : mName(utils::getElementValue<std::string>(element, {"Name"}).value())
    , mTitle(utils::getElementValue<std::string>(element, {"Title"}).value_or("Untitled"))
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
