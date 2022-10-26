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
      constructor(emitter, key, readonly) {
        super(key);
        this.component = {
          props: ['readonly', 'emitter', 'ikey', 'getData', 'putData'],
          template: '<input type="number" :readonly="readonly" :value="value" @input="change($event) " @dblclick.stop="" @pointerdown.stop="" @pointermove.stop=""/>',
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
              if (this.ikey)
                this.putData(this.ikey, this.value)
              this.emitter.trigger('process');
            }
          },
          mounted() {
            this.value = this.getData(this.ikey);
          }
        };
        this.props = { emitter, ikey: key, readonly };
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
        return node.addControl(new %NAME%Control(this.editor, 'number'))
                   .addOutput(output);
      }

      worker(node, inputs, outputs) {
        outputs['number'] = node.data.number;
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
