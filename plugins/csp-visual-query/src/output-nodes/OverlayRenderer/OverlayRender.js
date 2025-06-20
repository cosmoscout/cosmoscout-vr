////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The TimeNode is pretty simple as it only has a single output socket. The component serves as
// a kind of factory. Whenever a new node is created, the builder() method is called.
class OverlayRenderComponent extends Rete.Component {
  constructor() {
    // This name must match the RenderNode::sName defined in Render.cpp.
    super("OverlayRender");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Output";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {
    let input = new Rete.Input('Image2D', "Image 2D", CosmoScout.socketTypes['Image2D']);
    node.addInput(input);

    let lutInput = new Rete.Input('lut', "LookUp Table", CosmoScout.socketTypes['LUT']);
    node.addInput(lutInput);

    const minMaxInput = new Rete.Input('minMax', 'LUT Min/Max', CosmoScout.socketTypes['RVec2']);
    node.addInput(minMaxInput);

    let centerControl = new DropDownControl('center',
        (selection) =>
            CosmoScout.sendMessageToCPP({operation: "setCenter", center: selection.text}, node.id),
        "Body", [{value: 0, text: 'None'}]);
    node.addControl(centerControl);

    node.onMessageFromCPP = (message) => centerControl.setOptions(
        message.map((centerName, index) => ({value: index, text: centerName})));

    let opacityControl = new SliderControl('opacity',
        (value) => CosmoScout.sendMessageToCPP({operation: "setOpacity", opacity: value}, node.id),
        ()      => {}, "Opacity");
    node.addControl(opacityControl);

    node.onInit =
        (nodeDiv) => {
          console.log("OverlayRenderNode.onInit", node.id, node.data);
          centerControl.init(nodeDiv, {
            options: node.data.options?.map((body, index) => ({value: index, text: body})),
            selectedValue: node.data.selectedBody
          });
          opacityControl.init(nodeDiv, 0, 1, 0.01, node.data.opacity || 1.0);
        }

    return node;
  }
}
