////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "NumberNode.hpp"

#include "../../../../src/cs-utils/utils.hpp"

namespace csp::demonodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string NumberNode::getName() {
  return "Number";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string NumberNode::getSource() {
  std::string source = R"(
    class %NAME%Control extends Rete.Control {
      constructor(key) {
        super(key);

        this.template = `
          <input class="number-input" type="text" value="0" />

          <style>
            .number-input {
              margin: 10px 15px !important;
              width: 150px !important;
            }
          </style>
        `;
      }

      init(nodeElement) {
        const el = nodeElement.querySelector("input");
        el.addEventListener('change', (e) => {
          CosmoScout.sendMessagetoCPP(parseFloat(e.target.value), this.parent.id);
        });
      }
    }

    class %NAME%Component extends Rete.Component {
      constructor() {
        super("%NAME%");
        this.category = "Inputs";
      }

      builder(node) {
        let output = new Rete.Output('output', "Output", CosmoScout.socketTypes['Number Value']);
        node.addOutput(output);

        let control = new %NAME%Control('number');
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

std::unique_ptr<NumberNode> NumberNode::create() {
  return std::make_unique<NumberNode>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NumberNode::onMessageFromJS(nlohmann::json const& data) {
  double value = json;
  writeOutput("output", value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
