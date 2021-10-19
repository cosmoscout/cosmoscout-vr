////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WCS_OVERLAYS_WEB_MAP_EXCEPTION_HPP
#define CSP_WCS_OVERLAYS_WEB_MAP_EXCEPTION_HPP

#include <VistaTools/tinyXML/tinyxml.h>

#include <string>
#include <vector>

namespace csp::wcsoverlays {

/// Class to store a single WCS exception.
class WebCoverageException : public std::exception {
 public:
  /// Possible exception codes.
  /// Descriptions are taken from Table E.1 of the WCS 1.3.0 Implementation Specification.
  enum class Code {
    /// No code could be determined.
    eNone,
    /// Request contains a Format not offered by the server.
    eInvalidFormat,
    /// Request contains a CRS not offered by the server for one or more of the Layers in the
    /// request.
    eInvalidCRS,
    /// GetMap request is for a Layer not offered by the server, or GetFeatureInfo request is for a
    /// Layer not shown on the map.
    eLayerNotDefined,
    /// Request is for a Layer in a Style not offered by the server.
    eStyleNotDefined,
    /// GetFeatureInfo request is applied to a Layer which is not declared queryable.
    eLayerNotQueryable,
    /// GetFeatureInfo request contains invalid I or J value.
    eInvalidPoint,
    /// Value of (optional) UpdateSequence parameter in GetCapabilities request is equal to current
    /// value of service metadata update sequence number.
    eCurrentUpdateSequence,
    /// Value of (optional) UpdateSequence parameter in GetCapabilities request is greater than
    /// current value of service metadata update sequence number.
    eInvalidUpdateSequence,
    /// Request does not include a sample dimension value, and the server did not declare a default
    /// value for that dimension.
    eMissingDimensionValue,
    /// Request contains an invalid sample dimension value.
    eInvalidDimensionValue,
    /// Request is for an optional operation that is not supported by the server.
    eOperationNotSupported,
    /// General Error
    eNoApplicableCode,
  };

  explicit WebCoverageException(VistaXML::TiXmlElement* element);

  inline bool operator!=(const WebCoverageException& rhs) const {
    return mCode != rhs.mCode || mText != rhs.mText;
  }

  /// Get the code identifying the type of exception that occurred.
  Code getCode() const;
  /// Get a short description of the error.
  std::string const& getText() const;

  /// Get type and description of the error as one string.
  const char* what() const noexcept override;

 private:
  Code        mCode;
  std::string mText;
  std::string mMessage;
};

/// Class to store a collection of WCS exceptions.
class WebCoverageExceptionReport : public std::exception {
 public:
  /// Construct a WebCoverageExceptionReport from a XML document.
  explicit WebCoverageExceptionReport(VistaXML::TiXmlDocument doc);
  /// Construct a WebCoverageExceptionReport from a string containing a XML document.
  explicit WebCoverageExceptionReport(std::string const& xml);

  /// Gets a list of WCS exceptions that occurred.
  std::vector<WebCoverageException> const& getExceptions() const;

  /// Gets a single string describing all WCS exceptions that occurred.
  const char* what() const noexcept override;

 private:
  static VistaXML::TiXmlDocument parseXml(std::string const& xml);

  std::vector<WebCoverageException> mExceptions;
  std::string                       mMessage;
};

} // namespace csp::wcsoverlays

#endif // CSP_WCS_OVERLAYS_WEB_MAP_EXCEPTION_HPP
