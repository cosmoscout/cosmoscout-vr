////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_DEMO_NODE_EDITOR_MATH_NODE_HPP
#define CSP_DEMO_NODE_EDITOR_MATH_NODE_HPP

#include "../../../csl-node-editor/src/Node.hpp"

namespace csp::demonodeeditor {

///
class MathNode : public csl::nodeeditor::Node {
 public:
  enum class Operation { eAdd, eSubtract, eMultiply, eDivide };

  static const std::string         NAME;
  static const std::string         SOURCE;
  static std::unique_ptr<MathNode> create();

  std::string const& getName() const override;

  void process() override;

  void onMessageFromJS(nlohmann::json const& message) override;

  nlohmann::json getData() const override;
  void           setData(nlohmann::json const& json) override;

 private:
  Operation mOperation = Operation::eAdd;
};

} // namespace csp::demonodeeditor

#endif // CSP_DEMO_NODE_EDITOR_MATH_NODE_HPP
