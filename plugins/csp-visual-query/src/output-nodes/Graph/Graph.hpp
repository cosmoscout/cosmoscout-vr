////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_GRAPH_NODE_HPP
#define CSP_VISUAL_QUERY_GRAPH_NODE_HPP

#include "../../../../csl-node-editor/src/Node.hpp"

namespace csp::visualquery {

/// The Graph displays information about a selected coverage.
class Graph : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string      sName;
  static std::string            sSource();
  static std::unique_ptr<Graph> sCreate();

  // instance interface ----------------------------------------------------------------------------

  /// Each node must override this. It simply returns the static sName.
  std::string const& getName() const override;

  /// This simply updates the displayed information of the connected coverage. This method may also
  /// get called occasionally by the node editor, for example if a new web client was connected
  /// hence needs updated values for all nodes.
  void process() override;

  /// This is called whenever the node needs to be serialized. It returns a JSON object containing
  /// the current value.
  nlohmann::json getData() const override;

  /// This is called whenever the node needs to be deserialized. The given JSON object should
  /// contain a number under the key "value". the current value.
  void setData(nlohmann::json const& json) override;

 private:
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_GRAPH_NODE_HPP
