////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_TiME_INTERVAL_NODE_HPP
#define CSP_VISUAL_QUERY_TiME_INTERVAL_NODE_HPP

#include "../../../../csl-node-editor/src/Node.hpp"
#include "../../../../csl-ogc/src/common/utils.hpp"
#include "../../../../../src/cs-core/TimeControl.hpp"

namespace csp::visualquery {

/// The TimeInterval provides a user defined date and time. It demonstrates how a node can provide data
/// defined by the user. The node will write a new value to its output whenever the user edits the
/// the value and thus trigger a reprocessing of the connected nodes. The current value is stored as a
/// private member and will be serialized and deserialized whenever the node graph is saved or
/// loaded.
class TimeInterval : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string           sName;
  static std::string                 sSource();
  static std::unique_ptr<TimeInterval> sCreate(std::shared_ptr<cs::core::TimeControl> timeControl);

  // instance interface ----------------------------------------------------------------------------

  explicit TimeInterval(std::shared_ptr<cs::core::TimeControl> timeControl);
  ~TimeInterval();

  /// Each node must override this. It simply returns the static sName.
  std::string const& getName() const override;

  /// Whenever the user changes the value value of the node, the TimeInterval will send a message to
  /// the C++ instance of the node via onMessageFromJS, which in turn will call the process()
  /// method. This simply updates the value of the 'value' output. This method may also get called
  /// occasionally by the node editor, for example if a new web client was connected hence needs
  /// updated values for all nodes.
  void process() override;

  /// This will be called whenever the CosmoScout.sendMessageToCPP() is called by the JavaScript
  /// client part of this node.
  /// @param message  A JSON object as sent by the JavaScript node. In this case, it is actually
  ///                 just the currently selected value.
  void onMessageFromJS(nlohmann::json const& message) override;

  /// This is called whenever the node needs to be serialized. It returns a JSON object containing
  /// the current value.
  nlohmann::json getData() const override;

  /// This is called whenever the node needs to be deserialized. The given JSON object should
  /// contain a number under the key "value". the current value.
  void setData(nlohmann::json const& json) override;

 private:

  nlohmann::json createIntervalsMessage() const;

  std::shared_ptr<cs::core::TimeControl> mTimeControl;
  std::string mValue;
  std::vector<csl::ogc::TimeInterval> mIntervals;
  int mSelectedIntervalIndex;
  int mTimeOperationCounter;
  int mMaxTimeOperationCounter;
  bool mSyncSimTime;
  int mTimeConnection;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_TiME_INTERVAL_NODE_HPP