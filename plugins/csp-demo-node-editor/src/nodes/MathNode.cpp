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

std::string MathNode::getName() {
  return "Math";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string MathNode::getSource() {
  std::string source = R"(
    class %NAME%Control extends Rete.Control {
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

      init(nodeElement) {
        const el = nodeElement.querySelector("select");
        $(el).selectpicker();
        el.addEventListener('change', (e) => {
          CosmoScout.sendMessagetoCPP(parseInt(e.target.value), this.parent.id);
        });
      }
    }

    class %NAME%Component extends Rete.Component {
      constructor() {
        super("%NAME%");
        this.category = "Operations";
      }

      builder(node) {
        let first = new Rete.Input('first', "First", CosmoScout.socketTypes['Number Value']);
        node.addInput(first);

        let second = new Rete.Input('second', "Second", CosmoScout.socketTypes['Number Value']);
        node.addInput(second);

        let output = new Rete.Output('result', "Result", CosmoScout.socketTypes['Number Value']);
        node.addOutput(output);

        let control = new %NAME%Control('select');
        node.addControl(control);

        node.onInit = (nodeElement) => {
          control.init(nodeElement);
        };

        return node;
      }
    }
  )";

  cs::utils::replaceString(source, "%NAME%", getName());

  return source;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<MathNode> MathNode::create() {
  return std::make_unique<MathNode>();
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

void MathNode::onMessageFromJS(nlohmann::json const& data) {
  mOperation = data;
  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
