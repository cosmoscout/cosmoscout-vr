////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WebCoverage.hpp"
#include "../logger.hpp"
#include "WebCoverageException.hpp"

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

namespace csl::ogc {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverage::WebCoverage(VistaXML::TiXmlElement* element, Settings settings, std::string mUrl)
    : mUrl(std::move(mUrl))
    , mSettings(std::move(settings)) {
  auto title = utils::getElementValue<std::string>(element, {"ows:Title"});
  auto id    = utils::getElementValue<std::string>(element, {"wcs:CoverageId"});

  mId    = id.value_or("");
  mTitle = title.has_value() ? title.value() : "No Title";

  mAbstract = utils::getElementValue<std::string>(element, {"ows:Abstract"});

  auto* keywordList = element->FirstChild("ows:Keywords");

  if (keywordList) {
    for (auto* keyword = keywordList->FirstChild("ows:Keyword"); keyword;
         keyword       = keyword->NextSibling()) {

      auto content = utils::getElementValue<std::string>(keyword->ToElement());
      if (content.has_value() && !content.value().empty()) {
        mKeywords.push_back(utils::getElementValue<std::string>(keyword->ToElement()).value());
      }
    }
  }

  auto* wgs84BB = element->FirstChildElement("ows:WGS84BoundingBox");
  if (!wgs84BB) {
    throw std::runtime_error("Layer is missing BoundingBox.");
  }

  auto* lower = wgs84BB->FirstChildElement("ows:LowerCorner");
  auto* upper = wgs84BB->FirstChildElement("ows:UpperCorner");

  std::vector<std::string> lowerSplit =
      utils::split(utils::getElementValue<std::string>(lower).value(), ' ');
  std::vector<std::string> upperSplit =
      utils::split(utils::getElementValue<std::string>(upper).value(), ' ');

  if (lowerSplit.size() != 2 || upperSplit.size() != 2) {
    throw std::runtime_error("Could not parse bounds");
  }

  mSettings.mBounds.mMinLon = std::stod(lowerSplit[0]);
  mSettings.mBounds.mMinLat = std::stod(lowerSplit[1]);

  mSettings.mBounds.mMaxLon = std::stod(upperSplit[0]);
  mSettings.mBounds.mMaxLat = std::stod(upperSplit[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WebCoverage::getTitle() const {
  return mTitle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WebCoverage::getId() const {
  return mId;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::string> const& WebCoverage::getAbstract() const {
  return mAbstract;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::string> WebCoverage::getKeywords() const {
  if (mKeywords.empty()) {
    return {};
  }

  std::stringstream ss;
  for (size_t i = 0; i < mKeywords.size(); ++i) {
    if (i != 0) {
      ss << ", ";
    }
    ss << mKeywords[i];
  }
  std::string s = ss.str();

  return s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverage::Settings const& WebCoverage::getSettings() const {
  return mSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebCoverage::isRequestable() const {
  return strlen(mId.c_str()) > 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebCoverage::loadCoverageDetails() {
  std::stringstream urlStream;
  urlStream << mUrl;
  urlStream << "?SERVICE=WCS";
  urlStream << "&VERSION=2.0.1";
  urlStream << "&REQUEST=DescribeCoverage";
  urlStream << "&COVERAGEID=" << mId;

  std::stringstream xmlStream;
  curlpp::Easy      request;
  request.setOpt(curlpp::options::Url(urlStream.str()));
  request.setOpt(curlpp::options::WriteStream(&xmlStream));
  request.setOpt(curlpp::options::NoSignal(true));
  request.setOpt(curlpp::options::SslVerifyPeer(false));

  try {
    request.perform();
  } catch (std::exception const& e) {
    std::stringstream message;
    message << "WCS describe coverage request failed for '" << mUrl << "': '" << e.what() << "'";
    throw std::runtime_error(message.str());
  }

  std::string             docString = xmlStream.str();
  VistaXML::TiXmlDocument doc;
  doc.Parse(docString.c_str());
  if (doc.Error()) {
    std::stringstream message;
    message << "Parsing WCS coverage description failed for '" << mUrl << "': '" << doc.ErrorDesc()
            << "'";
    throw std::runtime_error(message.str());
  }

  // Check for WCS exception
  try {
    WebCoverageExceptionReport e(doc);
    std::stringstream          message;
    message << "Requesting WCS capabilities for '" << mUrl << "' resulted in WCS exception: '"
            << e.what() << "'";
    throw std::runtime_error(message.str());
  } catch (std::exception const&) {
    // No WCS exception occurred
  }

  mDoc.emplace(doc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebCoverage::parseTime() {
  auto* time = mDoc.value()
                   .FirstChildElement("wcs:CoverageDescriptions")
                   ->FirstChildElement("wcs:CoverageDescription")
                   ->FirstChildElement("gmlcov:metadata")
                   ->FirstChildElement("gmlcov:Extension")
                   ->FirstChildElement("wcsgs:TimeDomain");

  if (time && time->FirstChildElement("gml:TimePeriod")) {
    time        = time->FirstChildElement("gml:TimePeriod");
    auto* begin = time->FirstChildElement("gml:beginPosition");
    auto* end   = time->FirstChildElement("gml:endPosition");

    if (begin && end) {
      auto startTime = utils::getElementValue<std::string>(begin);
      auto endTime   = utils::getElementValue<std::string>(end);

      if (startTime.has_value() && endTime.has_value()) {
        mSettings.mTimeIntervals.clear();
        std::stringstream timeString;
        timeString << startTime.value();
        timeString << "/";
        timeString << endTime.value();

        /// This creates an ISO 8601 timeduration composed of a start and end point
        /// defined in <gml:beginPosition> and <gml:endPosition>
        auto* period = time->FirstChildElement("gml:timeInterval");
        if (period) {
          auto periodUnit    = utils::getAttribute<std::string>(period, "unit");
          auto periodContent = utils::getElementValue<std::string>(period);

          if (periodUnit.has_value() && periodContent.has_value()) {
            std::stringstream periodString;
            periodString << "P";
            if (periodUnit.value() == "year") {
              periodString << periodContent.value() << "Y";
            } else if (periodUnit.value() == "month") {
              periodString << periodContent.value() << "M";
            } else if (periodUnit.value() == "day") {
              periodString << periodContent.value() << "D";
            } else if (periodUnit.value() == "hour") {
              periodString << "T";
              periodString << periodContent.value() << "H";
            } else if (periodUnit.value() == "minute") {
              periodString << "T";
              periodString << periodContent.value() << "M";
            } else if (periodUnit.value() == "second") {
              periodString << "T";
              periodString << periodContent.value() << "S";
            }

            if (periodString.str() != "P") {
              timeString << "/" << periodString.str();
            }
          }
        }

        try {
          utils::parseIsoString(timeString.str(), mSettings.mTimeIntervals);
        } catch (std::exception const& e) {
          logger().warn("Failed to parse Iso String '{}' for layer '{}': '{}'. No time dependent "
                        "data will be available for this layer!",
              timeString.str(), mTitle, e.what());
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebCoverage::update() {
  loadCoverageDetails();
  parseTime();
  parseDetails();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebCoverage::parseDetails() {
  if (mSettings.mAxisLabels.size() == 2) {
    return;
  }

  auto* labels = mDoc.value()
                     .FirstChildElement("wcs:CoverageDescriptions")
                     ->FirstChildElement("wcs:CoverageDescription")
                     ->FirstChildElement("gml:domainSet")
                     ->FirstChildElement("gml:RectifiedGrid")
                     ->FirstChildElement("gml:axisLabels");

  auto labelsValue = utils::getElementValue<std::string>(labels);

  if (labelsValue.has_value()) {
    std::vector<std::string> labelsSplit = utils::split(labelsValue.value(), ' ');
    if (labelsSplit.size() == 2) {
      mSettings.mAxisLabels = labelsSplit;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::ogc
