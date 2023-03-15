////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The TimeNode is pretty simple as it only has a single output socket. The component serves as
// a kind of factory. Whenever a new node is created, the builder() method is called.
class RenderComponent extends Rete.Component {
  constructor() {
    // This name must match the RenderNode::sName defined in Render.cpp.
    super("Render");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Output";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single input. The first parameter is the name of this input and must be
    // unique amongst all sockets. It is also used in the RenderNode::process() to read the
    // input of this node. The second parameter is shown as name on the node. The last parameter
    // references a socket type which has been registered with the node factory before.
    let input = new Rete.Input('grey-scale-geo-texture', "Map Overlay", CosmoScout.socketTypes['GreyScaleGeoTexture']);
    node.addInput(input);


    // Add the number display. The name parameter must be unique amongst all controls of this
    // node. The RenderControl class is defined further below.
    let heatLegendControl = new RenderControl('render');
    node.addControl(heatLegendControl);

    // Whenever a message from C++ arrives, we set the input value accordingly. This message is
    // sent by the RenderNode::process() method.
    node.onMessageFromCPP = (message) => {
      heatLegendControl.setValue(message.value);
    };

    setTimeout(() => heatLegendControl.setValue([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 5000)

    return node;
  }
}

// This is the widget which is used for rendering the data.
class RenderControl extends Rete.Control {

  constructor(key) {
    super(key);

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `<div id="csp-visual-query.tfEditor"></div>`;
  }

  // This is called by the node.onMessageFromCPP method above whenever a new value is sent in
  // from C++.
  setValue(val) {
    if (!this.transferFunctionEditor) {
      this.transferFunctionEditor = CosmoScout.transferFunctionEditor.create(
        document.getElementById("csp-visual-query.tfEditor"),
        (tf) => this._transferFunction = tf,
        {
          width: 450,
          height: 120,
          defaultFunction: "HeatLight.json",
          fitToData: true,
          numberBins: 32
        });
    }
    // Each node container gets the id "#node-<id>". This way we can select elements inside the
    // node using a selector. Here we select the p element with the class "display-value" as
    // defined by the template above.
    this.transferFunctionEditor.setData(val);
  }

  getValue() {
    return this._transferFunction;
  }
}
  
  