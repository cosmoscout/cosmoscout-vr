////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_DEMO_NODE_EDITOR_TIME_NODE_HPP
#define CSP_DEMO_NODE_EDITOR_TIME_NODE_HPP

#include "../../../csl-node-editor/src/NodeEditor.hpp"

namespace cs::core {
class TimeControl;
}

namespace csp::demonodeeditor {

///
class TimeNode : public csl::nodeeditor::Node {
 public:
  static std::string getName();
  static std::string getSource();

  static std::unique_ptr<TimeNode> create(std::shared_ptr<cs::core::TimeControl> pTimeControl);

  TimeNode(std::shared_ptr<cs::core::TimeControl> pTimeControl);
  ~TimeNode() override;

 private:
  std::shared_ptr<cs::core::TimeControl> mTimeControl;
  int                                    mTimeConnection = 0;
};

} // namespace csp::demonodeeditor

#endif // CSP_DEMO_NODE_EDITOR_TIME_NODE_HPP
