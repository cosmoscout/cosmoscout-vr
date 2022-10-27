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
      constructor(editor) {

        const key = "number";

        super(key);
        this.component = {
          props: ['editor', 'key', 'getData', 'putData'],
          template: `<input type="number" :value="value" @input="change($event) " 
                            @dblclick.stop="" @pointerdown.stop="" @pointermove.stop="" />`,
          data() {
            return {
              value: 0,
            }
          },
          methods: {
            change(e) {
              this.value = +e.target.value;
              this.update();
            },
            update() {
              this.putData(this.key, this.value);
              console.log(this.key);
              this.editor.trigger('process');
            }
          },
          mounted() {
            this.value = this.getData(this.key);
          }
        };
        this.props = { editor, key };
      }

      setValue(val) {
        this.vueContext.value = val;
      }
    }

    class %NAME%Component extends Rete.Component {
      constructor() {
        super("%NAME%");

        this.category = "Inputs";
      }

      builder(node) {
        let output = new Rete.Output('number', "Number", CosmoScout.socketTypes['Number Value']);
        return node.addControl(new %NAME%Control(this.editor))
                   .addOutput(output);
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

} // namespace csp::demonodeeditor
