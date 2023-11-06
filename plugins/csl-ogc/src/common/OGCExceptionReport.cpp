////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "OGCExceptionReport.hpp"

#include <spdlog/fmt/fmt.h>

namespace csl::ogc {

////////////////////////////////////////////////////////////////////////////////////////////////////

OGCExceptionReport::OGCExceptionReport(std::vector<std::unique_ptr<OGCException>> exceptions)
    : mExceptions(std::move(exceptions)) {
  if (mExceptions.empty()) {
    mMessage = "No OGC exceptions found.";
  } else if (mExceptions.size() == 1) {
    mMessage = mExceptions[0]->what();
  } else {
    std::stringstream message;
    message << "Multiple OGC exceptions occurred: ";
    for (auto const& e : mExceptions) {
      message << fmt::format("'{}'", e->what());
      if (e != mExceptions.back()) {
        message << ", ";
      }
    }
    mMessage = message.str();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::unique_ptr<OGCException>> const&
OGCExceptionReport::getExceptions() const noexcept {
  return mExceptions;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* OGCExceptionReport::what() const noexcept {
  return mMessage.c_str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaXML::TiXmlDocument OGCExceptionReport::parseXml(std::string const& xml) {
  VistaXML::TiXmlDocument doc;
  doc.Parse(xml.c_str());
  if (doc.Error()) {
    throw std::runtime_error(fmt::format("Parsing XML failed: {}", doc.ErrorDesc()));
  }
  return doc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::ogc
