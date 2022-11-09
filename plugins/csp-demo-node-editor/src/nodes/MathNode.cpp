////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "MathNode.hpp"

#include "../logger.hpp"

#include "../../../../src/cs-utils/utils.hpp"

namespace csp::demonodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string MathNode::NAME = "Math";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string MathNode::SOURCE = R"(
    class MathControl extends Rete.Control {
      constructor(key) {
        super(key);

        this.template = `
          <select>
            <option value="0">Add</option>
            <option value="1">Subtract</option>
            <option value="2">Multiply</option>
            <option value="3">Divide</option>
          </select>

          <style>
            .dropdown {
              margin: 10px 15px !important;
              width: 150px !important;
            }
          </style>
        `;
      }

      init(nodeDiv, data) {
        const el = nodeDiv.querySelector("select");
        $(el).selectpicker();

        if (data.operation) {
          $(el).selectpicker('val', data.operation);
        }

        el.addEventListener('change', (e) => {
          CosmoScout.sendMessagetoCPP(parseInt(e.target.value), this.parent.id);
        });
      }
    }

    class MathComponent extends Rete.Component {
      constructor() {
        super("Math");
        this.category = "Operations";
      }

      builder(node) {
        let first = new Rete.Input('first', "First", CosmoScout.socketTypes['Number Value']);
        node.addInput(first);

        let second = new Rete.Input('second', "Second", CosmoScout.socketTypes['Number Value']);
        node.addInput(second);

        let output = new Rete.Output('result', "Result", CosmoScout.socketTypes['Number Value']);
        node.addOutput(output);

        let control = new MathControl('select');
        node.addControl(control);

        node.onInit = (nodeDiv) => {
          control.init(nodeDiv, node.data);
        };

        return node;
      }
    }
  )";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<MathNode> MathNode::create() {
  return std::make_unique<MathNode>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& MathNode::getName() const {
  return NAME;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MathNode::process() {
  double first  = readInput<double>("first", 0.0);
  double second = readInput<double>("second", 0.0);

  double result = 0.0;

  switch (mOperation) {
  case Operation::eAdd:
    result = first + second;
    break;
  case Operation::eSubtract:
    result = first - second;
    break;
  case Operation::eMultiply:
    result = first * second;
    break;
  case Operation::eDivide:
    result = first / second;
    break;
  default:
    break;
  }

  writeOutput("result", result);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MathNode::onMessageFromJS(nlohmann::json const& message) {
  mOperation = message;
  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json MathNode::getData() const {
  return {{"operation", mOperation}};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MathNode::setData(nlohmann::json const& json) {
  mOperation = json["operation"];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
