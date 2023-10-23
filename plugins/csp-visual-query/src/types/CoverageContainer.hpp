////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_COVERAGE_CONTAINER_HPP
#define CSP_VISUAL_QUERY_COVERAGE_CONTAINER_HPP

#include "../../../csl-ogc/src/wcs/WebCoverageService.hpp"

namespace csp::visualquery {

struct CoverageContainer {

  CoverageContainer(std::shared_ptr<csl::ogc::WebCoverageService> server, 
   std::shared_ptr<csl::ogc::WebCoverage> imageChannel)
    : mServer(std::move(server))
    , mImageChannel(std::move(imageChannel)) {
  }
  
  std::shared_ptr<csl::ogc::WebCoverageService> mServer;
  std::shared_ptr<csl::ogc::WebCoverage>        mImageChannel;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_COVERAGE_CONTAINER_HPP