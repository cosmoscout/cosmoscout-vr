////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

class JsonVolumeFileLoaderComponent extends Rete.Component {

  constructor() {
    super("JsonVolumeFileLoader");
    this.category = "Input";
  }

  builder(node) {
    let output = new Rete.Output('Volume3D', 'Volume 3D', CosmoScout.socketTypes['Volume3D']);
    node.addOutput(output)

    const textControl = new TextControl('FileLoader');
    node.addControl(textControl);

    node.onInit = (nodeDiv) => textControl.init(nodeDiv, node.data);

    return node;
  }
}

class TextControl extends Rete.Control {
  constructor(key) {
    super(key);

    this.data = {};

    this.id = crypto.randomUUID();

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
      <input type="text" id="text-input-${this.id}" />
    `;
  }

  init(nodeDiv, data) {
    const el = nodeDiv.querySelector(`#text-input-${this.id}`);

    if (data?.file) {
      el.value  = data.file;
      this.data = data;
    }

    el.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        this.data.file = el.value;
        CosmoScout.sendMessageToCPP({file: el.value}, this.parent.id);
      }
    });
  }
}