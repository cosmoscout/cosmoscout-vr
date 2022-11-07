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

          <style scoped>
            .dropdown {
              margin: 10px 15px !important;
              width: 150px !important;
            }
          </style>
        `;
      }

      setValue(val) {
        const el = document.querySelector("#node-" + this.parent.id + " .value");
        el.innerHTML = val;
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

        let select = new %NAME%Control('select');
        node.addControl(select);

        node.onInit = (nodeElement) => {
          select.init(nodeElement);
        };

        node.onMessageFromCPP = (message) => {
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

  if (hasNewInput()) {
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MathNode::onMessageFromJS(nlohmann::json const& json) {
  uint32_t value = json;

  mOperation = static_cast<Operation>(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
