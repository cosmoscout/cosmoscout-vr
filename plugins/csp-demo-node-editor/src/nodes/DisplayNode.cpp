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

std::string DisplayNode::getName() {
  return "Display";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string DisplayNode::getSource() {
  std::string source = R"(
    class %NAME%Control extends Rete.Control {
      constructor(key) {
        super(key);

        this.template = `
          <p class="value">0</p>

          <style scoped>
            p {
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
        const el = document.querySelector("#node-" + this.parent.id + " .value");
        el.innerHTML = val;
      }
    }

    class %NAME%Component extends Rete.Component {
      constructor() {
        super("%NAME%");
        this.category = "Outputs";
      }

      builder(node) {
        let input = new Rete.Input('number', "Number", CosmoScout.socketTypes['Number Value']);
        node.addInput(input);

        let control = new %NAME%Control('display');
        node.addControl(control);

        node.onInit = (element) => {
          console.log("init");
          console.log(element);
        };

        node.onMessage = (message) => {
          control.setValue(message.value);
        };

        return node;
      }
    }
  )";

  cs::utils::replaceString(source, "%NAME%", getName());

  return source;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<DisplayNode> DisplayNode::create() {
  return std::make_unique<DisplayNode>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::process() {

  if (hasNewInput()) {
    double value = readInput<double>("number", 0.0);

    nlohmann::json json;
    json["value"] = value;
    sendMessage(json);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
