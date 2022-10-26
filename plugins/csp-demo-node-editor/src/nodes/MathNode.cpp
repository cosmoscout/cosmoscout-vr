////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "MathNode.hpp"

namespace csp::demonodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string MathNode::getName() {
  return "Math";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string MathNode::getSource() {
  std::string source = R"(
     class NumControl extends Rete.Control {

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

    class NumComponent extends Rete.Component {

      constructor() {
        super("Number");
      }

      builder(node) {
        var out1 = new Rete.Output('num', "Number", numSocket);

        return node.addControl(new NumControl(this.editor, 'num')).addOutput(out1);
      }

      worker(node, inputs, outputs) {
        outputs['num'] = node.data.num;
      }
    }
  )";

  return source;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<MathNode> MathNode::create() {
  return std::make_unique<MathNode>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::demonodeeditor
