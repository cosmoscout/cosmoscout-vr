////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_DEMO_NODE_EDITOR_NUMBER_NODE_HPP
#define CSP_DEMO_NODE_EDITOR_NUMBER_NODE_HPP

#include "../../../csl-node-editor/src/Node.hpp"

namespace csp::demonodeeditor {

///
class NumberNode : public csl::nodeeditor::Node {
 public:
  static const std::string           NAME;
  static const std::string           SOURCE;
  static std::unique_ptr<NumberNode> create();

  std::string const& getName() const override;

  void process() override;

  void onMessageFromJS(nlohmann::json const& message) override;

  nlohmann::json getData() const override;
  void           setData(nlohmann::json const& json) override;

 private:
  double mValue = 0.0;
};

} // namespace csp::demonodeeditor

#endif // CSP_DEMO_NODE_EDITOR_NUMBER_NODE_HPP
