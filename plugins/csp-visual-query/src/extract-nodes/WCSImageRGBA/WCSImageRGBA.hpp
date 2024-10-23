////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_WCS_IMAGE_RGBA_HPP
#define CSP_VISUAL_QUERY_WCS_IMAGE_RGBA_HPP

#include "../../../../csl-node-editor/src/Node.hpp"
#include "../../../../csl-ogc/src/wcs/WebCoverageService.hpp"
#include "../../../../csl-ogc/src/wcs/WebCoverageTextureLoader.hpp"
#include "../../types/types.hpp"

namespace csp::visualquery {

class WCSImageRGBA : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string             sName;
  static std::string                   sSource();
  static std::unique_ptr<WCSImageRGBA> sCreate();

  // instance interface ----------------------------------------------------------------------------

  /// Each node must override this. It simply returns the static sName.
  std::string const& getName() const override;

  /// Whenever the simulation time changes, the TimeNode will call this method itself. It simply
  /// updates the value of the 'time' output. This method may also get called occasionally by the
  /// node editor, for example if a new web client was connected hence needs updated values for all
  /// nodes.
  void process() override;

  // Creates a new request object to load a texture from a server
  csl::ogc::WebCoverageTextureLoader::Request getRequest();
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_WCS_IMAGE_RGBA_HPP