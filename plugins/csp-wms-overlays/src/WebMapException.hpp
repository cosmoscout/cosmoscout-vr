////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WMS_OVERLAYS_WEB_MAP_EXCEPTION_HPP
#define CSP_WMS_OVERLAYS_WEB_MAP_EXCEPTION_HPP

#include <VistaTools/tinyXML/tinyxml.h>

#include <string>
#include <vector>

namespace csp::wmsoverlays {

/// Class to store a single WMS exception.
class WebMapException : public std::exception {
 public:
  /// Possible exception codes.
  /// Descriptions are taken from Table E.1 of the WMS 1.3.0 Implementation Specification.
  enum class Code {
    eNone,          ///< No code could be determined.
    eInvalidFormat, ///< Request contains a Format not offered by the server.
    eInvalidCRS, ///< Request contains a CRS not offered by the server for one or more of the Layers
                 ///< in the request.
    eLayerNotDefined,       ///< GetMap request is for a Layer not offered by the server, or
                            ///< GetFeatureInfo request is for a Layer not shown on the map.
    eStyleNotDefined,       ///< Request is for a Layer in a Style not offered by the server.
    eLayerNotQueryable,     ///< GetFeatureInfo request is applied to a Layer which is not declared
                            ///< queryable.
    eInvalidPoint,          ///< GetFeatureInfo request contains invalid I or J value.
    eCurrentUpdateSequence, ///< Value of (optional) UpdateSequence parameter in GetCapabilities
                            ///< request is equal to current value of service metadata update
                            ///< sequence number.
    eInvalidUpdateSequence, ///< Value of (optional) UpdateSequence parameter in GetCapabilities
                            ///< request is greater than current value of service metadata update
                            ///< sequence number.
    eMissingDimensionValue, ///< Request does not include a sample dimension value, and the server
                            ///< did not declare a default value for that dimension.
    eInvalidDimensionValue, ///< Request contains an invalid sample dimension value.
    eOperationNotSupported, ///< Request is for an optional operation that is not supported by the
                            ///< server.
  };

  WebMapException(VistaXML::TiXmlElement* element);

  inline bool operator!=(const WebMapException& rhs) const {
    return mCode != rhs.mCode || mText != rhs.mText;
  }

  /// Get the code identifying the type of exception that occurred.
  Code getCode() const;
  /// Get a short description of the error.
  std::string const& getText() const;

  /// Get type and description of the error as one string.
  virtual const char* what() const noexcept;

 private:
  Code        mCode;
  std::string mText;
  std::string mMessage;
};

/// Class to store a collection of WMS exceptions.
class WebMapExceptionReport : public std::exception {
 public:
  /// Construct a WebMapExceptionReport from a XML document.
  WebMapExceptionReport(VistaXML::TiXmlDocument doc);
  /// Construct a WebMapExceptionReport from a string containing a XML document.
  WebMapExceptionReport(std::string const& xml);

  /// Gets a list of WMS exceptions that occurred.
  std::vector<WebMapException> const& getExceptions() const;

  /// Gets a single string describing all WMS exceptions that occurred.
  virtual const char* what() const noexcept;

 private:
  VistaXML::TiXmlDocument parseXml(std::string const& xml);

  std::vector<WebMapException> mExceptions;
  std::string                  mMessage;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_OVERLAYS_WEB_MAP_EXCEPTION_HPP
