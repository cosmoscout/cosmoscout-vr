////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OWS_EXCEPTION_REPORT
#define CSL_OGC_OWS_EXCEPTION_REPORT

#include "csl_ogc_export.hpp"

#include <optional>
#include <string>
#include <vector>

namespace ogc::schemas::ows {

struct ExceptionType;

struct CSL_OGC_EXPORT ExceptionReportType {
  // attributes
  std::string                version;
  std::optional<std::string> lang;

  // child elements
  std::vector<ExceptionType> exceptions;
};

struct CSL_OGC_EXPORT ExceptionType {
  // attributes
  std::string                exceptionCode;
  std::optional<std::string> locator;

  // child elements
  std::vector<std::string> exceptionTexts;
};

} // namespace ogc::schemas::ows

#endif // CSL_OGC_OWS_EXCEPTION_REPORT