////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

import {TransferFunctionEditor} from "../../third-party/js/transfer-function-editor.js";

class TransferFunctionComponent extends Rete.Component {
  constructor() {
    super("TransferFunction");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Input";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {
    const tfControl = new TransferFunctionControl("volume-tf");

    const lutOutput = new Rete.Output('lut', 'LookUp Table', CosmoScout.socketTypes['LUT']);
    node.addOutput(lutOutput);

    node.addControl(tfControl);

    const tfCallback = (newLut) => CosmoScout.sendMessageToCPP({lut: newLut}, node.id)

    node.onInit = (nodeDiv) => { tfControl.init(nodeDiv, tfCallback); };
    return node;
  }
}

class TransferFunctionControl extends Rete.Control {
  constructor(key) {
    super(key);

    this.id       = crypto.randomUUID();
    this.template = `
      <style>
        .tfe-transparency-editor {
            height: 200px;
            border: 1px solid black;
        }
        .tfe-color-map-editor {
            margin-top: 10px;
  
            height: 40px;
            border: 1px solid black;
        }
        .tfe-color-map-editor-color-picker-container {
            background-color: var(--cs-color-background-dark) !important;
        }
        .tfe-color-map-editor-bin-selector-checkbox {
          display: inline !important;
        }
      </style>
      <div 
        id="tf-editor-root-${this.id}"
        class="tf-editor-root container-fluid"
        style="height: 300px" 
      />
    `
  }

  init(nodeDiv, callback, data) {
    this.el = nodeDiv.querySelector(`#tf-editor-root-${this.id}`);
    this.el.addEventListener("pointerdown", (e) => {e.stopPropagation()});
    this.el.addEventListener("wheel", (e) => e.stopPropagation());

    this.tf = new TransferFunctionEditor(this.el);

    const dd = this.el.querySelector(".tfe-color-map-editor-interpolation-method-select");
    dd.classList.add("form-select", "dropdown-toggle", "btn");
    dd.style.textAlign = "left";

    dd.querySelectorAll("option").forEach(
        el => { el.classList.add("dropdown-item", "dropdown-menu", "inner", "show"); });

    let first = true;

    this.tf.addListener((tfEditor) => {
      const lut = tfEditor.createLookUpTable(256).map((value) => hexToRGBA(value));
      if (!first) {
        callback(lut);
      } else {
        first = false;
      }
    });
  }
}

function getChunksFromString(string, chunkSize) {
  return string.match(new RegExp(`.{${chunkSize}}`, "g"));
}

function hexToRGBA(hex) {
  const chunkSize = Math.floor((hex.length - 1) / 3);
  const hexArr    = getChunksFromString(hex.slice(1), chunkSize);
  return hexArr.map((hexString) => parseInt(hexString.repeat(2 / hexString.length), 16) / 255);
}