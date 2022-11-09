////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "DisplayNode.hpp"

#include "../logger.hpp"

#include "../../../../src/cs-utils/utils.hpp"

namespace csp::demonodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string DisplayNode::NAME = "Display";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string DisplayNode::SOURCE = R"(
    class DisplayControl extends Rete.Control {
      constructor(key) {
        super(key);

        this.template = `
          <p class="display-value">0</p>

          <style>
            p.display-value {
              font-family: 'Ubuntu Mono', monospace;
              border-radius: var(--cs-border-radius-medium);
              background: rgba(255, 255, 255, 0.1);
              width: 200px;
              padding: 5px 15px;
              margin: 10px;
              text-align: right;
              font-size: 1.1em;
            }
          </style>
        `;
      }

      setValue(val) {
        const el = document.querySelector("#node-" + this.parent.id + " .display-value");
        el.innerHTML = val;
      }
    }

    class DisplayComponent extends Rete.Component {
      constructor() {
        super("Display");
        this.category = "Outputs";
      }

      builder(node) {
        let input = new Rete.Input('number', "Number", CosmoScout.socketTypes['Number Value']);
        node.addInput(input);

        let control = new DisplayControl('display');
        node.addControl(control);

        node.onInit = (element) => {
        };

        node.onMessageFromCPP = (message) => {
          control.setValue(message.value);
        };

        return node;
      }
    }
  )";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<DisplayNode> DisplayNode::create() {
  return std::make_unique<DisplayNode>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& DisplayNode::getName() const {
  return NAME;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::process() {
  nlohmann::json json;
  json["value"] = readInput<double>("number", 0.0);
  sendMessageToJS(json);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
