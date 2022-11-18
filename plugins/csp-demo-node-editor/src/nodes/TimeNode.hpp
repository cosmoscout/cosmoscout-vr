////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_DEMO_NODE_EDITOR_TIME_NODE_HPP
#define CSP_DEMO_NODE_EDITOR_TIME_NODE_HPP

#include "../../../csl-node-editor/src/Node.hpp"

namespace cs::core {
class TimeControl;
}

namespace csp::demonodeeditor {

/// The TimeNode provides the current simulation time of CosmoScout VR in seconds. It demonstrates
/// how a node can provide new data whenever something in CosmoScout VR changes. The node will write
/// a new value to its output whenever the simulation time changes and thus trigger a reprocessing
/// of the connected nodes.
class TimeNode : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string         sName;
  static std::string               sSource();
  static std::unique_ptr<TimeNode> sCreate(std::shared_ptr<cs::core::TimeControl> pTimeControl);

  // instance interface ----------------------------------------------------------------------------

  /// New instances of this node are created by the node factory.
  /// @param timeControl This is used to get the current simulation time of CosmoScout VR.
  explicit TimeNode(std::shared_ptr<cs::core::TimeControl> timeControl);
  ~TimeNode() override;

  /// Each node must override this. It simply returns the static sName.
  std::string const& getName() const override;

  /// Whenever the simulation time changes, the TimeNode will call this method itself. It simply
  /// updates the value of the 'time' output. This method may also get called occasionally by the
  /// node editor, for example if a new web client was connected hence needs updated values for all
  /// nodes.
  void process() override;

 private:
  std::shared_ptr<cs::core::TimeControl> mTimeControl;
  int                                    mTimeConnection = 0;
};

} // namespace csp::demonodeeditor

#endif // CSP_DEMO_NODE_EDITOR_TIME_NODE_HPP
