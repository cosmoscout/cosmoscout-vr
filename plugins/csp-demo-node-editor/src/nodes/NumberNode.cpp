////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "NumberNode.hpp"

#include "../../../../src/cs-utils/utils.hpp"

namespace csp::demonodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string NumberNode::NAME = "Number";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string NumberNode::SOURCE = R"(
    class NumberControl extends Rete.Control {
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

      init(nodeDiv, data) {
        const el = nodeDiv.querySelector("input");

        if (data.value) {
          el.value = data.value;
        }

        el.addEventListener('input', e => {
          CosmoScout.sendMessagetoCPP(parseFloat(e.target.value), this.parent.id);
        });
        el.addEventListener('pointermove', e => e.stopPropagation());
      }
    }

    class NumberComponent extends Rete.Component {
      constructor() {
        super("Number");
        this.category = "Inputs";
      }

      builder(node) {
        let output = new Rete.Output('output', "Output", CosmoScout.socketTypes['Number Value']);
        node.addOutput(output);

        let control = new NumberControl('number');
        node.addControl(control);

        node.onInit = (nodeDiv) => {
          control.init(nodeDiv, node.data);
        };

        return node;
      }
    }
  )";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<NumberNode> NumberNode::create() {
  return std::make_unique<NumberNode>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& NumberNode::getName() const {
  return NAME;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NumberNode::process() {
  writeOutput("output", mValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NumberNode::onMessageFromJS(nlohmann::json const& message) {
  mValue = message;
  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json NumberNode::getData() const {
  return {{"value", mValue}};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void NumberNode::setData(nlohmann::json const& json) {
  mValue = json["value"];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
