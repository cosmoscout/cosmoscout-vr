////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OGC_EXCEPTION_REPORT_HPP
#define CSL_OGC_OGC_EXCEPTION_REPORT_HPP

#include "OGCException.hpp"

#include <VistaTools/tinyXML/tinyxml.h>
#include <exception>
#include <memory>
#include <vector>

namespace csl::ogc {

class OGCExceptionReport : public std::exception {
 public:
  /// Gets a list of OGC exceptions that occurred.
  std::vector<std::unique_ptr<OGCException>> const& getExceptions() const noexcept;

  /// Gets a single string describing all OGC exceptions that occurred.
  const char* what() const noexcept override;

 protected:
  explicit OGCExceptionReport(std::vector<std::unique_ptr<OGCException>> exceptions);

  static VistaXML::TiXmlDocument parseXml(std::string const& xml);

 private:
  std::vector<std::unique_ptr<OGCException>> mExceptions{};
  std::string                                mMessage{};
};

} // namespace csl::ogc

#endif // CSL_OGC_OGC_EXCEPTION_REPORT_HPP