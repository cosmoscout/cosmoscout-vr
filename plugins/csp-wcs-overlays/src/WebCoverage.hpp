////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WCS_OVERLAYS_WEB_CAPABILITY_COVERAGE_HPP
#define CSP_WCS_OVERLAYS_WEB_CAPABILITY_COVERAGE_HPP

#include "utils.hpp"

#include <VistaTools/tinyXML/tinyxml.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace csp::wcsoverlays {
class WebCoverage {
 public:
  /// Struct for storing general layer settings.
  struct Settings {
    /// Default (maximum) bounds of the layer.
    Bounds mBounds;
    /// TimeIntervals, for which data is available.
    std::vector<TimeInterval> mTimeIntervals;
    /// Attribution for the layer.
    std::optional<std::string> mAttribution;
    /// Axis labels used for scaling
    std::vector<std::string> mAxisLabels;
  };

  WebCoverage(VistaXML::TiXmlElement* element, Settings settings, std::string mUrl);

  /// Gets a human readable description of the layer.
  std::string const& getTitle() const;
  /// Gets the id of the layer used for requests.
  std::string const& getId() const;
  /// Gets a narrative description of the layer.
  std::optional<std::string> const& getAbstract() const;
  /// Gets a narrative description of the layer.
  std::optional<std::string> getKeywords() const;
  /// Gets the general settings of the layer.
  Settings const& getSettings() const;

  /// Checks if map data may be requested for the layer.
  bool isRequestable() const;

  /// Re-requests the document found at mUrl and re-parses the time intervals
  /// Called from Plugin::setWCSCoverage
  void update();

 private:
  /// Makes a DescribeCoverage request
  void loadCoverageDetails();
  /// Parses time domain into TimeIntervals
  void parseTime();
  /// Parses coverage details into settings
  void parseDetails();

  /// URL to DescribeCoverage
  std::string                mUrl;
  std::string                mTitle;
  std::string                mId;
  std::optional<std::string> mAbstract;

  /// List of keywords for this coverage
  std::vector<std::string> mKeywords;

  /// DescribeCoverage Document
  std::optional<VistaXML::TiXmlDocument> mDoc;

  Settings mSettings;
};
} // namespace csp::wcsoverlays

#endif // CSP_WCS_OVERLAYS_WEB_CAPABILITY_COVERAGE_HPP
