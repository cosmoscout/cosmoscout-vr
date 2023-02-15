////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_WCS_SOURCE_HPP
#define CSP_VISUAL_QUERY_WCS_SOURCE_HPP

#include "../../../csl-node-editor/src/Node.hpp"

namespace csp::visualquery {

class WCSSource : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string         sName;
  static std::string               sSource();
  static std::unique_ptr<WCSSource> sCreate();

  // instance interface ----------------------------------------------------------------------------

  /// New instances of this node are created by the node factory.
  
  explicit WCSSource();
  ~WCSSource() override;

  /// Each node must override this. It simply returns the static sName.
  std::string const& getName() const override;

  /// Whenever the simulation time changes, the TimeNode will call this method itself. It simply
  /// updates the value of the 'time' output. This method may also get called occasionally by the
  /// node editor, for example if a new web client was connected hence needs updated values for all
  /// nodes.
  void process() override;

 private:
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_WCS_SOURCE_HPP