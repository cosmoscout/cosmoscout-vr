////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WCS_OVERLAYS_WEB_COVERAGE_EXCEPTION_HPP
#define CSP_WCS_OVERLAYS_WEB_COVERAGE_EXCEPTION_HPP

#include <VistaTools/tinyXML/tinyxml.h>

#include <string>
#include <vector>

namespace csp::wcsoverlays {

/// Class to store a single WCS exception.
class WebCoverageException : public std::exception {
 public:
  /// Possible exception codes.
  /// Descriptions are taken from 06-121r9 OGC Web Services Common Standard.
  /// https://www.scirp.org/(S(351jmbntvnsjt1aadkposzje))/reference/referencespapers.aspx?referenceid=1152641
  enum class Code {
    /// No code could be determined.
    eNone,
    /// One of the identifiers passed does not match with any of the coverages offered by this
    /// server
    eNoSuchCoverage,
    /// An empty list of identifiers was passed as input argument, while at least one identifier is
    /// required
    eEmptyCoverageIdList,
    /// The dimension subsetting operation specified an axis label that does not exist in
    /// the Envelope or has been used more than once in the GetCoverage request
    eInvalidAxisLabel,
    /// Operation request contains an invalid subsetting value; either a trim or slice parameter
    /// value is outside the extent of the coverage or, in a trim operation, a lower bound
    /// is above the upper bound
    eInvalidSubsetting,
    /// Operation request does not include a parameter value
    eMissingParameterValue,
    /// Operation request contains an invalid parameter value
    eInvalidParameterValue,
    /// List of versions in "AcceptVersions" parameter value,in GetCapabilities operation request,
    /// did not include any version supported by this server
    eVersionNegotiationFailed,
    /// Value of (optional) UpdateSequence parameter in GetCapabilities request is equal to current
    /// value of service metadata update sequence number.
    eCurrentUpdateSequence,
    /// Value of (optional) UpdateSequence parameter in GetCapabilities request is greater than
    /// current value of service metadata update sequence number.
    eInvalidUpdateSequence,
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

#endif // CSP_WCS_OVERLAYS_WEB_COVERAGE_EXCEPTION_HPP
